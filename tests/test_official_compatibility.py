import unittest

from ccb.official import (
    OFFICIAL_DATA_HASHES,
    load_official_records,
    verify_data_hash,
    verify_official_records,
)


class OfficialCompatibilityTests(unittest.TestCase):
    def test_all_fixed_records_match_exactly(self) -> None:
        for domain in OFFICIAL_DATA_HASHES:
            self.assertTrue(verify_data_hash(domain))
            records = load_official_records(domain)
            self.assertEqual(len(records), 400)
            report = verify_official_records(domain, records)
            self.assertTrue(report.compatible, "\n".join(report.mismatches[:10]))
            self.assertEqual(report.exact_matches, 400)


if __name__ == "__main__":
    unittest.main()

