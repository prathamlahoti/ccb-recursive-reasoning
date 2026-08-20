import unittest

from ccb.dataset import SplitConfig
from ccb.domains.symbolic_pointers import SymbolicPointersDomain
from ccb.domains.symbolic_pointers import PointerOperation
from ccb.structural import generate_filtered_split, operation_ngrams, structural_feature_audit
from ccb.structural_presets import build_structural_splits


class StructuralSplitTests(unittest.TestCase):
    def test_filtered_split_is_deterministic(self) -> None:
        domain = SymbolicPointersDomain()
        config = SplitConfig("heldout_pair", (3,), 4, 100)
        accept = lambda episode: PointerOperation.SET_D_TO_A_PLUS_B in episode.operations
        left = generate_filtered_split(domain.generate, config, accept)
        right = generate_filtered_split(domain.generate, config, accept)
        self.assertEqual(left, right)
        self.assertTrue(all(accept(episode) for episode in left))

    def test_feature_overlap_audit(self) -> None:
        domain = SymbolicPointersDomain()
        train = [domain.generate(depth=3, seed=1)]
        test = [domain.generate(depth=3, seed=2)]
        audit = structural_feature_audit(
            train, test, lambda episode: operation_ngrams(episode, 2)
        )
        self.assertIn("test_only_features", audit)

    def test_structural_presets_enforce_holdouts(self) -> None:
        for domain in ("d1", "d2", "d3"):
            splits, audit = build_structural_splits(domain, seeds_per_depth=1)
            self.assertGreater(len(splits["train"]), 0)
            self.assertEqual(audit.get("train_exposure", 0), 0)
            if "test_exposure" in audit:
                self.assertGreater(audit["test_exposure"], 0)


if __name__ == "__main__":
    unittest.main()
