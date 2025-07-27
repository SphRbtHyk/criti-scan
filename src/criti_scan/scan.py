"""Given an input pdf, perform the scanning of the critical edition.
"""
import os
import shutil
from pathlib import Path
from loguru import logger
import click
from criti_scan.extract_png import PNGExtracter
from criti_scan.detect_zones import ZoneDetecter
from criti_scan.ocr import OCRText


@click.command()
@click.argument('pdf_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(writable=True, dir_okay=False))
@click.option('--language', '-l', required=True, help='OCR language code (e.g., "eng", "grc").')
@click.option('--tessdata-dir', '-t', required=True, type=click.Path(exists=True, file_okay=False),
              help='Path to the directory containing the tessdata.')
@click.option('--rm-temp-dir/--keep-temp-dir', default=True,
              help='Whether to remove intermediate files after OCR (default: remove).')
def scan(pdf_path: str,
         language: str,
         tessdata_dir: str,
         output_path: str,
         rm_temp_dir: bool = True):
    """Given the path to a PDF file, perform the extraction of the images and
    perform the OCRing.

    Args:
        pdf_path (str): The path to the PDF to OCR.
        language (str): The language to use for the OCR.
        tessdata_dir (str): The path to the directory containing the tessdata.
        output_path (str): The path to write down the OCR content.
        rm_temp_dir (bool, defaults to True): Whether or not intermediate data
            should be removed.
    """
    logger.info(f"Process PDF in {pdf_path}")
    # Perform the image extraction
    png_extracter = PNGExtracter(pdf_path=pdf_path)
    png_extracter.extract()

    outputs = []
    # For each of the image
    for img in os.listdir(png_extracter.output_folder):
        # Detect the zone
        zone = ZoneDetecter(
            str(Path(png_extracter.output_folder)/img)).select_main_text()
        # Perform the OCR on the selected zone
        ocr = OCRText(zone,
                      language=language,
                      tessdata_dir=tessdata_dir).run_ocr()

        # Concatenate the output
        outputs.append(ocr)
    # Clean-up the temp directory
    if rm_temp_dir:
        shutil.rmtree(png_extracter.output_folder)

    # Write the output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(" ".join(outputs))
        logger.info(f"Wrote down output in {output_path}")


if __name__ == "__main__":
    scan()
