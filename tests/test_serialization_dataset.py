import json
import tempfile
import unittest
from pathlib import Path

from ccb.dataset import (
    SplitConfig,
    audit_disjoint_splits,
    build_manifest,
    generate_split,
    rehash_manifest,
    write_jsonl,
)
from ccb.domains.alien_grid import AlienGridDomain
from ccb.serialization import (
    SerializationError,
    parse_prediction,
    serialize_episode,
    serialize_prediction,
)


class SerializationTests(unittest.TestCase):
    def test_episode_serialization_is_stable(self) -> None:
        episode = AlienGridDomain().generate(depth=5, seed=1)
        left = serialize_episode(episode)
        right = serialize_episode(episode)
        self.assertEqual(left, right)
        payload = json.loads(left)
        self.assertEqual(payload["depth"], 5)
        self.assertEqual(payload["domain"], "alien_grid")

    def test_rehash_covers_added_fields(self) -> None:
        manifest = {"schema": "test", "manifest_hash": "stale"}
        first = rehash_manifest(manifest)
        second = rehash_manifest({**first, "new_field": 1})
        self.assertNotEqual(first["manifest_hash"], second["manifest_hash"])
        self.assertEqual(second, rehash_manifest(second))

    def test_prediction_round_trip(self) -> None:
        text = serialize_prediction([[1], [2]], [2])
        parsed = parse_prediction(text, expected_depth=2)
        self.assertEqual(parsed["answer"], [2])

    def test_strict_prediction_parser_rejects_extra_fields(self) -> None:
        with self.assertRaises(SerializationError):
            parse_prediction('{"trace":[],"answer":0,"comment":"repair me"}', expected_depth=0)

    def test_strict_prediction_parser_rejects_wrong_depth(self) -> None:
        with self.assertRaises(SerializationError):
            parse_prediction('{"trace":[],"answer":0}', expected_depth=1)


class DatasetTests(unittest.TestCase):
    def test_split_and_manifest_are_deterministic(self) -> None:
        domain = AlienGridDomain()
        train_config = SplitConfig("train", (2, 3), 3, 100)
        test_config = SplitConfig("test", (4, 5), 3, 500)
        splits = {
            "train": generate_split(domain.generate, train_config),
            "test": generate_split(domain.generate, test_config),
        }
        audit = audit_disjoint_splits(splits)
        self.assertTrue(audit["instances_disjoint"])
        self.assertTrue(audit["programs_disjoint"])
        left = build_manifest(splits, config={"version": "test"})
        right = build_manifest(splits, config={"version": "test"})
        self.assertEqual(left, right)

    def test_jsonl_writer(self) -> None:
        episodes = generate_split(
            AlienGridDomain().generate, SplitConfig("train", (2,), 2, 0)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            write_jsonl(path, episodes)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["schema"], "ccb_episode_v1")


if __name__ == "__main__":
    unittest.main()
