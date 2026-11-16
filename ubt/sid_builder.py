"""CPU OPQ/PQ semantic IDs fitted exclusively on static catalog features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def array_sha256(array):
    array = np.ascontiguousarray(array)
    header = json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True)
    h = hashlib.sha256(header.encode())
    h.update(array.tobytes())
    return h.hexdigest()


def load_features(path, key="text"):
    """Read row-ordered .npy vectors or a named array in a .npz asset."""
    path = Path(path)
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if key not in z:
                raise ValueError(f"The input archive has no {key!r} array")
            values = z[key]
    else:
        raise ValueError("Expected .npy or .npz feature arrays")
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Features must be a nonempty finite [items, dimensions] matrix")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Every catalog item must have a finite nonzero feature vector")
    return values


def quantization_input(features, normalize):
    values = np.array(features, dtype=np.float32, order="C", copy=True)
    if normalize:
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def fit_encoder(features, config):
    """Fit either separate OPQ/PQ or the full OPQ,IVF1,PQ transform chain."""
    import faiss

    m = int(config["codebooks"])
    n, dim = features.shape
    if m < 1 or int(config["bits_per_code"]) != 8 or dim % m or n < 256:
        raise ValueError("8-bit PQ requires at least 256 items and dimension divisible by codebooks")
    if int(config["threads"]) < 1:
        raise ValueError("CPU thread count must be positive")
    faiss.omp_set_num_threads(int(config["threads"]))
    values = quantization_input(features, config["normalize_quantization_input"])
    seed = int(config["seed"])
    if config["recipe"] == "static_opq_pq":
        opq = faiss.OPQMatrix(dim, m)
        opq.niter = int(config["opq_iterations"])
        opq.niter_pq_0 = int(config["first_pq_iterations"])
        opq.niter_pq = int(config["inner_pq_iterations"])
        opq.max_train_points = n
        inner = faiss.ProductQuantizer(dim, m, 8)
        inner.cp.seed = seed
        opq.pq = inner
        opq.train(values)
        pq = faiss.ProductQuantizer(dim, m, 8)
        pq.cp.seed = seed
        pq.cp.niter = int(config["final_pq_iterations"])
        pq.train(np.ascontiguousarray(opq.apply_py(values)))
        return {"opq": opq, "pq": pq}
    if config["recipe"] != "opq_ivf_pq":
        raise ValueError(f"Unknown SID recipe: {config['recipe']}")
    index = faiss.index_factory(dim, f"OPQ{m},IVF1,PQ{m}x8", faiss.METRIC_INNER_PRODUCT)
    opq = faiss.downcast_VectorTransform(index.chain.at(0))
    opq.niter = int(config["opq_iterations"])
    opq.niter_pq_0 = int(config["first_pq_iterations"])
    opq.niter_pq = int(config["inner_pq_iterations"])
    inner = faiss.ProductQuantizer(dim, m, 8)
    inner.cp.seed = seed
    inner.cp.niter = int(config["inner_pq_iterations"])
    opq.pq = inner
    ivf = faiss.downcast_index(index.index)
    ivf.cp.seed = seed
    ivf.pq.cp.seed = seed
    ivf.pq.cp.niter = int(config["final_pq_iterations"])
    if "min_points_per_centroid" in config:
        inner.cp.min_points_per_centroid = int(config["min_points_per_centroid"])
        ivf.pq.cp.min_points_per_centroid = int(config["min_points_per_centroid"])
    index.train(values)
    return {"index": index}


def encode_raw(encoder, features, config):
    """Use the complete trained transform, including its IVF residual origin."""
    import faiss

    values = quantization_input(features, config["normalize_quantization_input"])
    if "index" in encoder:
        index = encoder["index"]
        codes = index.sa_encode(values)
        transformed = values
        for i in range(index.chain.size()):
            transformed = faiss.downcast_VectorTransform(index.chain.at(i)).apply_py(transformed)
        ivf = faiss.downcast_index(index.index)
        if ivf.by_residual:
            transformed = transformed - ivf.quantizer.reconstruct(0)
        pq = ivf.pq
    else:
        transformed = encoder["opq"].apply_py(values)
        pq = encoder["pq"]
        codes = pq.compute_codes(np.ascontiguousarray(transformed))
    m = int(config["codebooks"])
    if codes.shape != (len(values), m):
        raise ValueError("Expected exactly one byte per SID codebook")
    centers = faiss.vector_to_array(pq.centroids).reshape(m, 256, -1)
    return np.asarray(codes, dtype=np.uint8), transformed, centers


def disambiguate(codes, transformed, centers):
    """Resolve duplicates by the least-cost unused one-coordinate alternative.

    Reserve every original SID first, retain the first occurrence of a duplicate,
    and process later occurrences in item-row order with stable cost tie breaking.
    """
    result = codes.copy()
    used = {tuple(row) for row in codes}
    seen, changes = set(), []
    m = codes.shape[1]
    for row, code in enumerate(codes):
        key = tuple(code)
        if key not in seen:
            seen.add(key)
            continue
        distances = ((transformed[row].reshape(m, 1, -1) - centers) ** 2).sum(-1)
        delta = distances - distances[np.arange(m), code][:, None]
        delta[np.arange(m), code] = np.inf
        for flat in np.argsort(delta, axis=None, kind="stable"):
            digit, value = divmod(int(flat), 256)
            if not np.isfinite(delta[digit, value]):
                continue
            candidate = code.copy()
            candidate[digit] = value
            if tuple(candidate) not in used:
                result[row] = candidate
                used.add(tuple(candidate))
                changes.append({"row": row, "digit": digit, "old": int(code[digit]),
                                "new": value, "extra_subspace_squared_error": float(delta[digit, value])})
                break
        else:
            raise ValueError(f"No unused one-coordinate SID alternative for item row {row}")
    return result, changes


def assign_codes(encoder, features, config):
    codes, transformed, centers = encode_raw(encoder, features, config)
    raw_unique = len(np.unique(codes, axis=0))
    if config["collision_policy"] == "unique":
        codes, changes = disambiguate(codes, transformed, centers)
    elif config["collision_policy"] == "keep":
        changes = []
    else:
        raise ValueError("Collision policy must be 'keep' or 'unique'")
    return codes, {"raw_unique_sids": raw_unique, "unique_sids": len(np.unique(codes, axis=0)),
                   "changed_rows": changes}


def save_encoder(encoder, folder):
    import faiss

    folder = Path(folder)
    if "index" in encoder:
        faiss.write_index(encoder["index"], str(folder / "encoder.faiss"))
    else:
        faiss.write_VectorTransform(encoder["opq"], str(folder / "rotation.faiss"))
        faiss.write_ProductQuantizer(encoder["pq"], str(folder / "quantizer.faiss"))


def load_encoder(folder, config):
    import faiss

    folder = Path(folder)
    if config["recipe"] == "opq_ivf_pq":
        return {"index": faiss.read_index(str(folder / "encoder.faiss"))}
    return {"opq": faiss.read_VectorTransform(str(folder / "rotation.faiss")),
            "pq": faiss.read_ProductQuantizer(str(folder / "quantizer.faiss"))}
