""" Image Processing """

# pylint: disable=E1101

from typing import Tuple, List
from dataclasses import dataclass
from utils import Utils
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2 as cv

__version__ = "1.0.0"

__all__ = []

RGB = Tuple[int, int, int]


@dataclass
class ImageProcessing:
    """ImageProcessing"""

    image: Image = None
    draw: ImageDraw = None
    width: int = 0
    height: int = 0
    pixel_data: List[RGB] = None
    initial_image: Image = None

    def read_file(self, path: str) -> "ImageProcessing":
        """read_file"""

        self.image = Image.open(path)
        self.initial_image = self.image.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.width = self.image.size[0]
        self.height = self.image.size[1]
        self.pixel_data = self.image.load()

        return self

    def gray_shades(self) -> "ImageProcessing":
        """gray_shades"""

        image_array = np.array(self.image)
        gray_image = np.mean(image_array, axis=2, dtype=int)
        self.image = Image.fromarray(gray_image.astype(np.uint8))
        self.draw = ImageDraw.Draw(self.image)

        return self

    def sepia(self, depth=50) -> "ImageProcessing":
        """sepia"""

        for i in range(self.width):
            for j in range(self.height):
                a = self.pixel_data[i, j][0]
                b = self.pixel_data[i, j][1]
                c = self.pixel_data[i, j][2]
                s = (a + b + c) // 3
                a = s + depth * 2
                b = s + depth
                c = s

                a = min(s + depth * 2, 255)
                b = min(s + depth, 255)
                c = min(s, 255)

                self.draw.point((i, j), (a, b, c))

        return self

    def negative(self) -> "ImageProcessing":
        """negative"""

        img_array = np.array(self.image)
        negative_array = 255 - img_array
        self.image = Image.fromarray(negative_array)
        self.draw = ImageDraw.Draw(self.image)

        return self

    def noise(self, factor=100) -> "ImageProcessing":
        """noise"""

        img_array = np.array(self.image)
        noise = np.random.randint(-factor, factor + 1, img_array.shape, dtype=np.int16)
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self.image = Image.fromarray(noisy_img)
        self.draw = ImageDraw.Draw(self.image)

        return self

    def brightness(self, factor: int = 100) -> "ImageProcessing":
        """brightness"""

        img_array = np.array(self.image)
        bright_img = np.clip(img_array.astype(np.int16) + factor, 0, 255).astype(
            np.uint8
        )
        self.image = Image.fromarray(bright_img)
        self.draw = ImageDraw.Draw(self.image)

        return self

    def monochrome(self, factor: int = 100) -> "ImageProcessing":
        """monochrome"""

        img_array = np.array(self.image)
        sum_rgb = img_array.sum(axis=2)
        threshold = ((255 + factor) // 2) * 3
        mono_img = np.where(sum_rgb > threshold, 255, 0)
        mono_img = np.stack([mono_img] * 3, axis=-1)
        self.image = Image.fromarray(mono_img.astype(np.uint8))
        self.draw = ImageDraw.Draw(self.image)

        return self

    def contour(self) -> "ImageProcessing":
        """contour"""

        self.image = self.image.filter(ImageFilter.CONTOUR)

        return self

    def reset(self) -> "ImageProcessing":
        """contour"""

        self.image = self.initial_image.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.pixel_data = self.image.load()

        return self

    def _img_processing(self) -> "ImageProcessing":
        """image_processing"""

        gray = cv.cvtColor(np.array(self.image), cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (3, 3), 0)
        edged = cv.Canny(blurred, 10, 250)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        morph = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)

        img_data = cv.cvtColor(morph, cv.COLOR_GRAY2BGR)

        self.image = Image.fromarray(img_data)
        self.draw = ImageDraw.Draw(self.image)

        return self

    def _find_contours(self):
        """find_contours"""

        img_array = np.array(self.image)
        gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        return contours

    def img_recognition(self):
        """img_recognition"""

        self._img_processing()
        img_cont = self._find_contours()

        total = 0
        img_array = np.array(self.image)

        for c in img_cont:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                cv.drawContours(img_array, [approx], -1, (0, 255, 0), 4)
                total += 1

        self.image = Image.fromarray(cv.cvtColor(img_array, cv.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.image)

        print(f"Знайдено {total} сегмент(\u0430) прямокутних \u043e\u0431'єктів")

        return self


def test() -> None:
    """Test function"""

    win_name = "Window"
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)

    def show() -> None:
        """show"""

        image = np.array(img_proces.image)
        data = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        cv.imshow(win_name, data)
        cv.waitKey(0)

    img_proces = ImageProcessing()
    img_proces.read_file(Utils.get_path("./images/image.jpg"))

    show()

    img_proces.reset().gray_shades()
    show()

    img_proces.reset().sepia()
    show()

    img_proces.reset().negative()
    show()

    img_proces.reset().noise()
    show()

    img_proces.reset().brightness()
    show()

    img_proces.reset().monochrome()
    show()

    img_proces.reset().contour()
    show()

    img_proces.img_recognition()
    show()

    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
