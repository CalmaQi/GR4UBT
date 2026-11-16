"""Fit or replay catalog SID encoders without modifying packaged training inputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ubt"))

from sid_builder import (array_sha256, assign_codes, fit_encoder, load_encoder,
                         load_features, save_encoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("beauty", "cellphones", "ml1m", "lastfm"), required=True)
    parser.add_argument("--embeddings", type=Path, help="row-ordered .npy or .npz text features")
    parser.add_argument("--sid-embeddings", type=Path,
                        help="optional separate quantization vectors, with identical item-row order")
    parser.add_argument("--output", type=Path, help="new output directory (must not exist)")
    parser.add_argument("--encoder-dir", type=Path, help="replay a saved encoder instead of fitting")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sid.json")
    args = parser.parse_args()

    recipes = json.loads(args.config.read_text())
    recipe = recipes["datasets"][args.dataset]
    config = {k: v for k, v in recipes.items() if k != "datasets"} | recipe
    if args.encoder_dir:
        previous = json.loads((args.encoder_dir / "manifest.json").read_text())
        if previous["dataset"] != args.dataset:
            raise ValueError("The saved encoder belongs to a different dataset")
        config = previous["config"]
    source = ROOT / "data/processed" / args.dataset / recipe["feature_asset"]
    embeddings_path = args.embeddings or source
    text = load_features(embeddings_path)
    sid_input = load_features(args.sid_embeddings) if args.sid_embeddings else text
    if len(text) != len(sid_input):
        raise ValueError("Text and quantization features must have the same item rows")
    output = args.output or ROOT / "runs/sid" / args.dataset
    if output.exists():
        raise FileExistsError("Output already exists; choose a new --output directory")
    # Uniqueness resolution depends on the full ordered catalog. Replay is an
    # exact reconstruction only for the same catalog inputs used at fitting.
    input_hash = array_sha256(sid_input)
    if args.encoder_dir and previous["sid_input_sha256"] != input_hash:
        raise ValueError("Replay requires the same quantization features and item-row order")
    import faiss

    faiss.omp_set_num_threads(int(config["threads"]))
    start = time.monotonic()
    print(f"{args.dataset}: {len(text)} items, {sid_input.shape[1]}D SID input, {config['recipe']}", flush=True)
    encoder = load_encoder(args.encoder_dir, config) if args.encoder_dir else fit_encoder(sid_input, config)
    codes, statistics = assign_codes(encoder, sid_input, config)
    with np.load(source, allow_pickle=False) as z:
        packaged_codes = z["codes"]
    same_shape = codes.shape == packaged_codes.shape
    same_codes = same_shape and bool(np.array_equal(codes, packaged_codes))
    output.mkdir(parents=True, exist_ok=False)
    save_encoder(encoder, output)
    np.savez_compressed(output / "features.npz", text=text, codes=codes)
    manifest = {
        "dataset": args.dataset, "config": config,
        "items": len(text), "text_dimensions": text.shape[1],
        "sid_dimensions": sid_input.shape[1], "faiss_version": faiss.__version__,
        "numpy_version": np.__version__, "text_sha256": array_sha256(text),
        "sid_input_sha256": input_hash, "codes_sha256": array_sha256(codes),
        "row_contract": "row r represents item ID r+1; row order is preserved",
        "fit_population": "static catalog features only; no interactions or held-out labels",
        "replayed_encoder": bool(args.encoder_dir),
        "matches_packaged_codes": same_codes,
        "packaged_code_rows_equal": int(np.all(codes == packaged_codes, axis=1).sum()) if same_shape else None,
        "elapsed_seconds": time.monotonic() - start, **statistics,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "items": len(text),
                      "unique_sids": statistics["unique_sids"],
                      "matches_packaged_codes": same_codes,
                      "seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
