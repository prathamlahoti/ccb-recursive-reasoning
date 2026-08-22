import unittest

from ccb.dataset import SplitConfig
from ccb.domains.alien_grid import GridOperation
from ccb.domains.symbolic_pointers import SymbolicPointersDomain
from ccb.domains.symbolic_pointers import PointerOperation
from ccb.structural import (
    d1_transformation_signature,
    generate_filtered_split,
    operation_ngrams,
    structural_feature_audit,
)
from ccb.structural_presets import (
    build_d1_semantic_structural_splits,
    build_structural_splits,
)


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

    def test_d1_semantic_signature_catches_alternate_spelling(self) -> None:
        reserved = d1_transformation_signature(
            (GridOperation.ROTATE_90_CW, GridOperation.SHIFT_ROW_2_LEFT)
        )
        alternate = d1_transformation_signature(
            (
                GridOperation.TRANSPOSE_GRID,
                GridOperation.FLIP_HORIZONTAL,
                GridOperation.SHIFT_ROW_2_LEFT,
            )
        )
        self.assertEqual(reserved, alternate)

    def test_d1_semantic_structural_preset_is_transition_disjoint(self) -> None:
        splits, audit = build_d1_semantic_structural_splits(seeds_per_depth=1)
        self.assertTrue(audit["randomized_initial_states"])
        self.assertEqual(audit["semantic_segment_exposure"]["train"], 0)
        self.assertEqual(audit["semantic_segment_exposure"]["validation"], 0)
        self.assertTrue(audit["transition_audit"]["transition_disjoint"])
        self.assertGreater(len(splits["test_semantic_pair_depth"]), 0)


if __name__ == "__main__":
    unittest.main()
