""" Image Enhancer """

# pylint: disable=E1101

import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# from utils import Utils
# from PIL import Image, ImageDraw, ImageFilter

__version__ = "1.0.0"

__all__ = []

Image = List[Tuple[int, int, int]]


@dataclass
class ImageEnhancer:
    """ImageEnhancer"""

    file_path: str = None
    image = None
    cdf = None

    def get_file(self, path: str) -> "ImageEnhancer":
        """read_file"""

        stack = inspect.stack()
        caller_frame = stack[1]

        self.file_path = str(
            Path(caller_frame.filename).parent.joinpath(path).resolve()
        )

        self.image = cv.imread(self.file_path)

        return self

    def seve_file(self, prefix: str) -> None:
        """seve_file"""

        cv.imwrite(
            str(
                Path(self.file_path)
                .parent.joinpath(prefix + "_" + Path(self.file_path).name)
                .resolve()
            ),
            self.image,
        )

    def show_file(self) -> None:
        """show_file"""

        win_name = "Window"

        cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
        cv.imshow(win_name, self.image)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_hist(self) -> None:
        """show_hist"""

        plt.hist(self.image.ravel(), 256, [0, 256])
        plt.show()

    def show_alignment_hist(self) -> None:
        """show_alignment_hist"""

        img = cv.imread(self.file_path, 0)
        hist, _bins = np.histogram(img.flatten(), 256, [0, 256])
        self.cdf = hist.cumsum()
        cdf_normalized = self.cdf * hist.max() / self.cdf.max()
        plt.plot(cdf_normalized, color="b")
        plt.hist(img.flatten(), 256, [0, 256], color="r")
        plt.xlim([0, 256])
        plt.legend(("cdf", "histogram"), loc="upper left")
        plt.show()

    def sharpen_image(self) -> "ImageEnhancer":
        """Sharpen the image"""

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.image = cv.filter2D(self.image, -1, kernel)
        self.seve_file("sharpen")

        return self

    def denoise_image(self) -> "ImageEnhancer":
        """Denoise the image"""

        self.image = cv.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)
        self.seve_file("denoise")

        return self

    def adjust_brightness_contrast(self, brightness=0, contrast=0) -> "ImageEnhancer":
        """Adjust brightness and contrast"""

        beta = brightness
        alpha = contrast / 127 + 1

        self.image = cv.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        self.seve_file("abc")

        return self

    def apply_clahe(self) -> "ImageEnhancer":
        """apply_clahe"""

        lab = cv.cvtColor(self.image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv.merge((cl, a, b))
        self.image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
        self.seve_file("clahe")

        return self

    def show_result(self) -> "ImageEnhancer":
        """show_result"""

        cdf_m = np.ma.masked_equal(self.cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

        img = cv.imread(self.file_path, 0)
        equ = cv.equalizeHist(img)
        self.image = np.hstack((img, equ))
        self.seve_file("equalize")

        return self


def test() -> None:
    """Test function"""


if __name__ == "__main__":
    test()
