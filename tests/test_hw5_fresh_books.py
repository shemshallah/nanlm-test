"""HW5:0 fresh independent Gutenberg book source parsing and intake guards."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hw5_fresh_books import CANDIDATES, clean_plaintext, fetch_book


class FreshBookSourceTests(unittest.TestCase):
    def test_strips_license_and_normalizes_utf8_body(self):
        source = (
            b"header Gutenberg license\r\n"
            b"*** START OF THE PROJECT GUTENBERG EBOOK THE TITLE ***\r\n"
            + b"Alpha beta gamma.\r\n" * 100 +
            b"*** END OF THE PROJECT GUTENBERG EBOOK THE TITLE ***\r\nlicense"
        )
        content = clean_plaintext(source)
        self.assertTrue(content.startswith(b"Alpha beta gamma.\n"))
        self.assertTrue(content.endswith(b"Alpha beta gamma.\n"))
        self.assertNotIn(b"PROJECT GUTENBERG EBOOK", content)
        self.assertEqual(content.count(b"\n"), 100)
        self.assertGreater(len(content), 1000)
        self.assertNotIn(b"\r", content)

    def test_missing_boilerplate_is_not_accepted(self):
        with self.assertRaises(ValueError):
            clean_plaintext(b"ordinary prose " * 1000)
        with self.assertRaises(ValueError):
            clean_plaintext(b"\xff" * 2000)
        with self.assertRaises(ValueError):
            clean_plaintext(b"tiny")

    def test_accepts_this_variation_of_gutenberg_marker(self):
        text = (b"*** START OF THIS PROJECT GUTENBERG EBOOK FOO ***\n"
                + b"Reader text about events.\n" * 100
                + b"*** END OF THIS PROJECT GUTENBERG EBOOK FOO ***")
        self.assertTrue(clean_plaintext(text).startswith(b"Reader text"))

    def test_candidate_pool_fixed_without_duplicates(self):
        self.assertGreaterEqual(len(CANDIDATES), 8)
        self.assertEqual(len(CANDIDATES), len(set(CANDIDATES)))
        with self.assertRaises(ValueError):
            fetch_book(0)

if __name__ == "__main__":
    unittest.main()
