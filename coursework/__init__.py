"""Laboratory work 3"""

# pylint: disable=E1101, C0412, W0603

import sys
from pathlib import Path
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

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


def mouse_callback(event, x, y, _flags, _params):
    """Mouse callback"""

    if event == cv.EVENT_LBUTTONUP:
        print(x, y)


def test():
    """Test function"""

    win_name = "Window"

    effects = [
        "gray_shades",
        "serpia",
        "negative",
        "noise",
        "brightness",
        "monochrome",
        "contour",
    ]

    img_proces = ImageProcessing()
    img_proces.read_file(Utils.get_path("./img.jpg"))

    for i, effect in enumerate(effects):
        icon_proces = ImageProcessing()
        file = icon_proces.read_file(Utils.get_path("./img.jpg"))

        effect_method = getattr(icon_proces, effect)
        effect_method()

        icon_size = file.image.size

        # icon_width = int(img_proces.image.size[0] / len(effects))
        # icon_height = int(icon_width * (icon_size[1] / icon_size[0]))
        icon_height = int(icon_size[1] / np.round(len(effects) / 2))
        icon_width = int(icon_height * (icon_size[0] / icon_size[1]))

        resized_icon = file.image.resize((icon_width, icon_height))

        img_proces.image.paste(resized_icon, (i % 2 * icon_width, i // 2 * icon_height))

    image_data = np.array(img_proces.image)
    bgr_data = cv.cvtColor(image_data, cv.COLOR_RGB2BGR)

    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(win_name, mouse_callback)
    cv.imshow(win_name, bgr_data)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
