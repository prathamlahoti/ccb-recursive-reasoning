import unittest
from dataclasses import replace

from ccb.audits import alien_grid_reachability, dataset_shortcut_audit
from ccb.domains.alien_grid import (
    AlienGridDomain,
    GridOperation,
    OFFICIAL_OPERATIONS,
)
from ccb.domains.social_logic import SocialLogicDomain
from ccb.domains.symbolic_pointers import SymbolicPointersDomain
from ccb.validation import EpisodeValidationError, validate_episode


class AuditTests(unittest.TestCase):
    def test_d1_reachability_exposes_program_compression(self) -> None:
        levels = alien_grid_reachability(
            maximum_depth=8, operations=OFFICIAL_OPERATIONS
        )
        self.assertEqual(levels[0].distinct_endpoints, 1)
        self.assertEqual(levels[8].total_programs, 7**8)
        self.assertLess(levels[8].distinct_endpoints, levels[8].total_programs)
        self.assertGreater(levels[8].endpoint_compression, 1)

    def test_d1_involutions_collapse_endpoints(self) -> None:
        levels = alien_grid_reachability(
            maximum_depth=2,
            operations=(GridOperation.REVERSE_GRID, GridOperation.FLIP_HORIZONTAL),
        )
        self.assertLess(levels[2].distinct_endpoints, levels[2].total_programs)

    def test_shortcut_audit_counts_duplicate_programs(self) -> None:
        domain = AlienGridDomain(randomize_initial_state=False)
        episodes = tuple(domain.generate(depth=1, seed=seed) for seed in range(20))
        audit = dataset_shortcut_audit(episodes)
        self.assertEqual(audit["examples"], 20)
        self.assertLessEqual(audit["unique_programs"], 7)
        self.assertGreater(audit["duplicate_programs"], 0)


class ValidationTests(unittest.TestCase):
    def test_all_generated_domains_replay(self) -> None:
        for episode in (
            AlienGridDomain().generate(depth=20, seed=7),
            SymbolicPointersDomain().generate(depth=20, seed=7),
            SocialLogicDomain().generate(depth=20, seed=7),
        ):
            validate_episode(episode)

    def test_corrupt_final_state_is_rejected(self) -> None:
        episode = AlienGridDomain().generate(depth=2, seed=3)
        corrupted = replace(episode, final_state=episode.initial_state)
        with self.assertRaises(EpisodeValidationError):
            validate_episode(corrupted)


if __name__ == "__main__":
    unittest.main()
