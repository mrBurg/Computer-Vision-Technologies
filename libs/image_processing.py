""" Image Processing """

# pylint: disable=E1101

from typing import Tuple, List
from dataclasses import dataclass, field
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
    pixel_data: List[RGB] = field(default_factory=lambda: [])
    initial_image: Image = None

    def read_file(self, path: str) -> any:
        """read_file"""

        self.image = Image.open(path)
        self.initial_image = self.image.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.width = self.image.size[0]
        self.height = self.image.size[1]
        self.pixel_data = self.image.load()

        return self

    def gray_shades(self):
        """gray_shades"""

        image_array = np.array(self.image)
        gray_image = np.mean(image_array, axis=2, dtype=int)
        self.image = Image.fromarray(gray_image.astype(np.uint8))
        # gray_image_rgb = np.stack((gray_image,) * 3, axis=-1)
        # self.image = Image.fromarray(gray_image_rgb.astype(np.uint8))
        self.draw = ImageDraw.Draw(self.image)

        # for i in range(self.width):
        #     for j in range(self.height):
        #         a = self.pixel_data[i, j][0]
        #         b = self.pixel_data[i, j][1]
        #         c = self.pixel_data[i, j][2]
        #         s = (a + b + c) // 3

        #         self.draw.point((i, j), (s, s, s))

        return self

    def serpia(self, depth=50) -> None:
        """serpia"""

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

    def negative(self) -> None:
        """negative"""

        img_array = np.array(self.image)
        negative_array = 255 - img_array
        self.image = Image.fromarray(negative_array)
        self.draw = ImageDraw.Draw(self.image)

        # for i in range(self.width):
        #     for j in range(self.height):
        #         a = self.pixel_data[i, j][0]
        #         b = self.pixel_data[i, j][1]
        #         c = self.pixel_data[i, j][2]

        #         self.draw.point((i, j), (255 - a, 255 - b, 255 - c))

        return self

    def noise(self, factor=100) -> None:
        """noise"""

        img_array = np.array(self.image)
        noise = np.random.randint(-factor, factor + 1, img_array.shape, dtype=np.int16)
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self.image = Image.fromarray(noisy_img)
        self.draw = ImageDraw.Draw(self.image)

        # for i in range(self.width):
        #     for j in range(self.height):
        #         rand = random.randint(-factor, factor)

        #         a = self.pixel_data[i, j][0] + rand
        #         b = self.pixel_data[i, j][1] + rand
        #         c = self.pixel_data[i, j][2] + rand

        #         a = min(max(self.pixel_data[i, j][0] + rand, 0), 255)
        #         b = min(max(self.pixel_data[i, j][1] + rand, 0), 255)
        #         c = min(max(self.pixel_data[i, j][2] + rand, 0), 255)

        #         self.draw.point((i, j), (a, b, c))

        return self

    def brightness(self, factor: int = 100) -> None:
        """brightness"""

        img_array = np.array(self.image)
        bright_img = np.clip(img_array.astype(np.int16) + factor, 0, 255).astype(
            np.uint8
        )
        self.image = Image.fromarray(bright_img)
        self.draw = ImageDraw.Draw(self.image)

        # for i in range(self.width):
        #     for j in range(self.height):
        #         r, g, b = self.pixel_data[i, j]

        #         r = min(max(r + factor, 0), 255)
        #         g = min(max(g + factor, 0), 255)
        #         b = min(max(b + factor, 0), 255)

        #         self.draw.point((i, j), (r, g, b))

        return self

    def monochrome(self, factor: int = 100) -> None:
        """monochrome"""

        img_array = np.array(self.image)
        sum_rgb = img_array.sum(axis=2)
        threshold = ((255 + factor) // 2) * 3
        mono_img = np.where(sum_rgb > threshold, 255, 0)
        mono_img = np.stack([mono_img] * 3, axis=-1)
        self.image = Image.fromarray(mono_img.astype(np.uint8))
        self.draw = ImageDraw.Draw(self.image)

        # for i in range(self.width):
        #     for j in range(self.height):
        #         a = self.pixel_data[i, j][0]
        #         b = self.pixel_data[i, j][1]
        #         c = self.pixel_data[i, j][2]

        #         s = a + b + c
        #         if s > (((255 + factor) // 2) * 3):
        #             a, b, c = 255, 255, 255
        #         else:
        #             a, b, c = 0, 0, 0
        #         self.draw.point((i, j), (a, b, c))

        return self

    def contour(self) -> None:
        """contour"""

        self.image = self.image.filter(ImageFilter.CONTOUR)

        return self

    def reset(self) -> None:
        """contour"""

        self.image = self.initial_image.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.pixel_data = self.image.load()

        return self


def test():
    """Test function"""

    def show() -> None:
        """show"""

        win_name = "Window"
        image = np.array(img.image)
        data = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
        cv.imshow(win_name, data)
        cv.waitKey(0)
        cv.destroyAllWindows()

    img_proces = ImageProcessing()
    img = img_proces.read_file(Utils.get_path("./img.jpg"))

    show()

    img = img_proces.reset().gray_shades()
    show()

    img = img_proces.reset().serpia()
    show()

    img = img_proces.reset().negative()
    show()

    img = img_proces.reset().noise()
    show()

    img = img_proces.reset().brightness()
    show()

    img = img_proces.reset().monochrome()
    show()

    img = img_proces.reset().contour()
    show()


if __name__ == "__main__":
    test()
