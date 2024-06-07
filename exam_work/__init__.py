"""Exam work"""

# pylint: disable=E1101, C0412, W0603

import sys
from pathlib import Path

LIBS_PATH = Path.cwd().resolve()

sys.path.append(str(LIBS_PATH))

try:
    from image_enhancer import ImageEnhancer
except ImportError:
    from libs.image_enhancer import ImageEnhancer


def test() -> None:
    """Test function"""

    ie = ImageEnhancer()

    ie.get_file("./images/img.jpg").show_file()

    ie.show_hist()
    ie.show_alignment_hist()

    ie.sharpen_image().show_file()
    ie.denoise_image().show_file()
    ie.adjust_brightness_contrast(brightness=30, contrast=50).show_file()
    ie.apply_clahe().show_file()
    ie.show_result().show_file()


if __name__ == "__main__":
    test()
