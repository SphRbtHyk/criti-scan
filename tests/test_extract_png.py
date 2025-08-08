"""Tests that extracting the PNGs behaves as expected."""
import unittest
import shutil
import os
from pathlib import Path
from criti_scan.ocr.extract_png import PNGExtracter


DATA_PATH = Path(__file__).parent.resolve() / "test_data"


class TestPngExtracter(unittest.TestCase):
    """Tests that extracting the PNGs behaves as expected."""

    def setUp(self) -> None:
        self.extractor = PNGExtracter(
            pdf_path=str(DATA_PATH)+"/test_pdf.pdf",
            output_folder="test_png"
        )

    def test_extract(self):
        """Test that extracting the PNG behaves as expected.
        """
        self.extractor.extract()
        # Check that repository contains the right files
        self.assertCountEqual(
            os.listdir("test_png"),
            ["png_0.png", "png_1.png", "png_2.png"]
        )

    def tearDown(self) -> None:
        """Clean up the output of the test by removing the folder."""
        shutil.rmtree("test_png")

if __name__ == "__main__":
    unittest.main()