"""Laboratory work 8"""

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
    img_proces.read_file(Utils.get_path("./laboratory_work_8/img.jpg"))
    show()

    img_proces.img_recognition()
    img_proces.image.save(get_path("img_recognition.jpg"))
    show()

    img_proces.sepia().noise()
    img_proces.image.save(get_path("sepia_noise.jpg"))
    # cv.imwrite(
    #     str(get_path("sepia_noise.jpg")),
    #     cv.cvtColor(np.array(img_proces.image), cv.COLOR_RGB2BGR),
    # )
    img_proces.read_file(get_path("sepia_noise.jpg"))
    show()

    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
