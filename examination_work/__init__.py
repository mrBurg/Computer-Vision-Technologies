"""Examination work"""

# pylint: disable=E1101, C0412, W0603

import sys
from pathlib import Path
import numpy as np
import cv2 as cv

LIBS_PATH = Path.cwd().resolve()

sys.path.append(str(LIBS_PATH))

try:
    from utils import Utils
except ImportError:
    from libs.utils import Utils

try:
    from image_processing import ImageProcessing
except ImportError:
    from libs.image_processing import ImageProcessing


def get_path(path: str) -> str:
    """Get path"""

    return Path(__file__).parent.joinpath(path).resolve()


def test() -> None:
    """Test function"""

    win_name = "Window"
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)

    def show() -> None:
        """show"""

        cv.imshow(win_name, cv.cvtColor(np.array(img_proces.image), cv.COLOR_RGB2BGR))
        cv.waitKey(0)

    img_proces = ImageProcessing()
    img_proces.read_file(
        Utils.get_path(Path(__file__).parent.name + "/images/img.jpg")
    ).sepia().brightness(-100)
    img_proces.image.save(get_path("./images/sepia_brightness.jpg"))
    show()

    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
