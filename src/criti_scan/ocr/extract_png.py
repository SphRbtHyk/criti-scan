"""Module to extract the data from the pdf as pictures.
"""
from pathlib import Path
import os
from pdf2image import convert_from_path
from loguru import logger


class PNGExtracter:
    """Class to extract PNGs from an input PDFs.
    """
    def __init__(self,
                 pdf_path: str,
                 output_folder: str = "PNG",
                 dpi: int = 800):
        """Initialization of the class.

        Args:
            pdf_path (str): The path to the PDF. output_folder (str): The path
            to output the PNGs to.
            dpi (int, defaults 800): The DPI to extract the PDF to.
        """
        # Check if the input file is really a pdf
        if not Path(pdf_path).suffix == ".pdf":
            logger.error("Input file is not a pdf.")
            raise ValueError("Input file is not a PDF.")
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract(self) -> None:
        """Extract the content of the PDF into PNGS, following the requested
        output path.
        """
        pages = convert_from_path(self.pdf_path,
                                  self.dpi)

        for ix, page in enumerate(pages):
            page.save(f"{self.output_folder}/png_{ix}.png")

        logger.info(f"Dumped PNGS in {self.output_folder}")
