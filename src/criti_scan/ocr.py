"""Module for OCRing the critical edition."""

import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pytesseract
from typing import Optional, Tuple


class OCRText:
    """
    A class for preprocessing an image for OCR and extracting text using Tesseract.

    Attributes:
        image (np.ndarray): Original input image (BGR or grayscale).
    """

    def __init__(self,
                 image: np.ndarray,
                 language: str,
                 tessdata_dir: str,
                 versify: bool = False
                 ):
        """
        Initializes the OCRText class with a given image.

        Args:
            image (np.ndarray): Input image in BGR or grayscale format.
            language (str): The language of the OCRed text.
            tessdata_dir (str): The directory where the data for tesseract is
                located.
            versify (bool): Whether or not the text contains verses.
        """
        self.image = image
        self.language = language
        self.tessdata_dir = tessdata_dir
        self.versify = versify

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Converts a color image to grayscale if needed."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def _enhance_contrast(self, gray: np.ndarray, factor: float) -> np.ndarray:
        """Enhances the contrast of a grayscale image."""
        pil_img = Image.fromarray(gray)
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)

    def _apply_threshold(self, img: np.ndarray) -> np.ndarray:
        """Applies adaptive thresholding to the image."""
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

    def _denoise(self, img: np.ndarray, ksize: int) -> np.ndarray:
        """Applies median blur to reduce noise."""
        if ksize >= 3 and ksize % 2 == 1:
            return cv2.medianBlur(img, ksize)
        return img

    def _crop_margins(self, img: np.ndarray, margins: Tuple[int, int, int, int]) -> np.ndarray:
        """Crops margins from the image."""
        h, w = img.shape
        top, bottom, left, right = margins
        return img[top:h - bottom if bottom else h, left:w - right if right else w]

    def preprocess(self,
                   contrast_factor: float = 2.0,
                   blur_ksize: int = 3,
                   crop_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
                   show_image: bool = False,
                   output_path: Optional[str] = None) -> np.ndarray:
        """
        Preprocesses the image for OCR and stores the result internally.

        Args:
            contrast_factor (float): Contrast enhancement factor.
            blur_ksize (int): Kernel size for median blur.
            crop_margins (Tuple[int, int, int, int]): Margins to crop (top, bottom, left, right).
            show_image (bool): Whether to display the preprocessed image.
            output_path (Optional[str]): If provided, saves the processed image to this path.

        Returns:
            np.ndarray: The preprocessed binary image.
        """
        gray = self._to_grayscale(self.image)
        contrasted = self._enhance_contrast(gray, contrast_factor)
        thresh = self._apply_threshold(contrasted)
        denoised = self._denoise(thresh, blur_ksize)
        cropped = self._crop_margins(denoised, crop_margins)

        if show_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(cropped, cmap='gray')
            plt.axis('off')
            plt.title("Preprocessed Image for OCR")
            plt.show()

        if output_path:
            cv2.imwrite(output_path, cropped)

        return cropped

    @staticmethod
    def clean_text(text: str,
                   versify: bool = False,
                   removed_symbols:
                   list[str] = ["!", "-\n", "Γ", "-", "“", '"', "]","!"]) -> str:
        """Clean the text outputted by Tesseract.

        Args:
            text (str): The text to clean.
            versity (bool, defaults to False): If set to True, then split the
                text on numbers.
            removed_symbols (list[str]): The symbols to remove from the text.
        """
        #TODO: remove -\n symbols
        # Remove symbols
        cleaned_text = "".join(
            [letter for letter in text if letter not in removed_symbols])
        # Remove the linebreaks
        cleaned_text = cleaned_text.replace("\n", " ")
        # Split on numbers if versify is set to true
        if versify:
            cleaned_text = "\n".join(re.split(r'(\d+)', cleaned_text))
        return cleaned_text

    def run_ocr(self,
                output_path: Optional[str] = None,
                preprocess_img: bool = False) -> str:
        """
        Runs Tesseract OCR on the preprocessed image.

        Args:
            output_path (str, Optional): If specified, where to write the
                output of the tesseract extraction.
            preprocess_img (bool, defaults to False): Whether or not the image
                should be pre-processed.
                
        Returns:
            str: The OCR-extracted text.
        """
        # Whitelist number -c tessedit_char_unblacklist=0123456789
        config = f'--tessdata-dir {self.tessdata_dir}'
        if preprocess_img:
            image = self.preprocess()
        else:
            image = self.image
        string_output = pytesseract.image_to_string(image,
                                                    lang=self.language,
                                                    config=config)
        output_text = self.clean_text(string_output,
                                      versify=self.versify)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(
                    self.clean_text(string_output)
                )
        return output_text
