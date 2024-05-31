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


def test() -> None:
    """Test function"""

    win_name = "Window"

    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    output_path = Path(__file__).parent.joinpath("final_img.jpg").resolve()

    img_proces = ImageProcessing()

    img_proces.read_file(Utils.get_path("./laboratory_work_8/img.jpg"))
    bgr_data = cv.cvtColor(np.array(img_proces.image), cv.COLOR_RGB2BGR)

    cv.imshow(win_name, bgr_data)
    cv.waitKey(0)

    img_proces.sepia().noise()
    img_proces.image.save(output_path)

    img_proces.read_file(output_path)
    bgr_data = cv.cvtColor(np.array(img_proces.image), cv.COLOR_RGB2BGR)

    cv.imshow(win_name, bgr_data)
    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
