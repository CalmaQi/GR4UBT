"""Exercise collision semantics and persisted FAISS encoding on CPU."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ubt"))
from sid_builder import (assign_codes, disambiguate, encode_raw, fit_encoder,
                         load_encoder, load_features, save_encoder)


class CollisionTests(unittest.TestCase):
    def test_first_occurrence_and_reserved_original_sids(self):
        codes = np.array([[0, 0], [0, 0], [1, 0], [0, 0]], dtype=np.uint8)
        centers = np.zeros((2, 256, 1), dtype=np.float32)
        centers[:, :, 0] = np.arange(256)
        transformed = np.zeros((4, 2), dtype=np.float32)
        original = codes.copy()
        result, changes = disambiguate(codes, transformed, centers)
        np.testing.assert_array_equal(codes, original)
        np.testing.assert_array_equal(result[[0, 2]], original[[0, 2]])
        self.assertEqual(len(np.unique(result, axis=0)), 4)
        self.assertEqual([x["row"] for x in changes], [1, 3])
        # (1,0) was reserved for an original item before processing duplicates.
        np.testing.assert_array_equal(result[1], [0, 1])
        np.testing.assert_array_equal(result[3], [2, 0])

    def test_invalid_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npy"
            for invalid in (np.array([[np.nan, 1]]), np.zeros((1, 2)), np.ones(4)):
                np.save(path, invalid)
                with self.assertRaises(ValueError):
                    load_features(path)


class EncoderTests(unittest.TestCase):
    def test_full_encoder_and_serialization_roundtrip(self):
        import faiss

        rng = np.random.default_rng(2024)
        features = rng.normal(size=(300, 16)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        # Exact feature duplicates exercise the post-quantization collision path.
        features[-2:] = features[0]
        config = dict(seed=2024, threads=1, codebooks=2, bits_per_code=8,
                      normalize_quantization_input=False, opq_iterations=1,
                      first_pq_iterations=1, inner_pq_iterations=1,
                      final_pq_iterations=1, min_points_per_centroid=1,
                      collision_policy="unique")
        for recipe in ("static_opq_pq", "opq_ivf_pq"):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as temporary:
                cfg = config | {"recipe": recipe}
                encoder = fit_encoder(features, cfg)
                raw, _, _ = encode_raw(encoder, features, cfg)
                if recipe == "opq_ivf_pq":
                    expected = encoder["index"].sa_encode(features)
                else:
                    expected = encoder["pq"].compute_codes(encoder["opq"].apply_py(features))
                np.testing.assert_array_equal(raw, expected)
                codes, report = assign_codes(encoder, features, cfg)
                self.assertEqual(codes.shape, (300, 2))
                self.assertEqual(codes.dtype, np.uint8)
                self.assertEqual(report["unique_sids"], 300)
                self.assertGreaterEqual(len(report["changed_rows"]), 2)
                save_encoder(encoder, temporary)
                restored = load_encoder(temporary, cfg)
                replay, replay_report = assign_codes(restored, features, cfg)
                np.testing.assert_array_equal(codes, replay)
                self.assertEqual(report, replay_report)
        self.assertTrue(faiss.__version__)


if __name__ == "__main__":
    unittest.main()
