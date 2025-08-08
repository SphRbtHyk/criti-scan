"""Module for detecting the different zones of the pdf."""
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from loguru import logger


class ZoneDetecter:
    """Given an image of a critical edition detect the different zones of the
    image.
    """

    def __init__(self,
                 img_path: str) -> None:
        """Initialize a ZoneDetector object.

        Args:
            img_path (str): Path to the image to consider.
        """
        self.image_path = img_path
        self.src = self._load_image()
        self.gray = self._convert_to_grayscale()
        self.thresh = self._apply_threshold()
        self.dilated = self._dilate_image()
        self.contours = self._find_contours()
        self.boxes = []

    def detect_zones(self):
        """Detect the different zones in the image.
        """
        self.output = self._draw_bounding_boxes()
        return self._draw_bounding_boxes()

    def _load_image(self) -> np.ndarray:
        """
        Loads the image from the file path.

        Returns:
            np.ndarray: The loaded BGR image.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """
        image = cv.imread(self.image_path)
        if image is None:
            logger.error(
                f"Could not open image: {self.image_path}"
            )
            raise FileNotFoundError(
                f"Could not open image: {self.image_path}")
        return image

    def _convert_to_grayscale(self) -> np.ndarray:
        """
        Converts the loaded image to grayscale.

        Returns:
            np.ndarray: Grayscale image.
        """
        return cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)

    def _apply_threshold(self) -> np.ndarray:
        """
        Applies adaptive thresholding to the grayscale image.

        Returns:
            np.ndarray: Binary image after adaptive thresholding.
        """
        return cv.adaptiveThreshold(
            self.gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY_INV, 15, 10
        )

    def _dilate_image(self) -> np.ndarray:
        """
        Applies morphological dilation to the thresholded image to
        connect nearby text.

        Returns:
            np.ndarray: Dilated image.
        """
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 120))
        return cv.dilate(self.thresh, kernel, iterations=1)

    def _find_contours(self):
        """
        Finds contours in the dilated image.

        Returns:
            list[np.ndarray]: Contours detected in the image.
        """
        contours, _ = cv.findContours(self.dilated,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
        return contours

    def _draw_bounding_boxes(self) -> np.ndarray:
        """
        Draws bounding boxes around detected contours and stores box
        information.

        Returns:
            np.ndarray: Image with bounding boxes drawn.
        """
        output = self.src.copy()
        for cnt in self.contours:
            x, y, w, h = cv.boundingRect(cnt)
            area = w * h
            if w > 50 and h > 50:  # Filter small areas (noise)
                cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                self.boxes.append({'box': (x, y, w, h), 'area': area})
        return output

    def display(self, figsize: tuple[int, int] = (12, 12)):
        """Display the image and the selected output as a rectangle zone.

        Args:
            figsize (tuple[int, int]): The size of the figure to consider.
        """
        output_rgb = cv.cvtColor(self.output,
                                 cv.COLOR_BGR2RGB)
        plt.figure(figsize=figsize)
        plt.imshow(output_rgb)
        plt.axis('off')
        plt.show()

    def select_main_text(self,
                         show: bool = False,
                         n_largest: int = 2):
        """Select the critical text by selecting the text with the highest
        y and among the n largest boxes of the page.

        Args:
            show (bool, defaults to False): whether or not to show the output
                and the corresponding bounding boxes.
            n_largest (int, defaults to 2): The number of boxes one must sort
                through.
        """
        self.detect_zones()

        # Sort by area, keep top 2
        top_boxes = sorted(self.boxes, key=lambda b: b['area'],
                           reverse=True)[:n_largest]

        # Select the topmost box (smallest y)
        selected = min(top_boxes, key=lambda b: b['box'][1])
        x, y, w, h = selected['box']

        # Draw red rectangle around selected block
        output = self.src.copy()
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 8)

        if show:
            output_rgb = cv.cvtColor(output, cv.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 12))
            plt.imshow(output_rgb)
            plt.title("Selected Centered Block")
            plt.axis('off')
            plt.show()

        # Return cropped block
        return self.src[y:y + h, x:x + w]
