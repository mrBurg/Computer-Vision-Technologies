"""Laboratory work 3"""

# pylint: disable=E1101, W0603

import sys
from pathlib import Path
from random import random, choice
import numpy as np
import cv2 as cv


LIBS_PATH = Path.cwd().resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory import Config, Utils, Polyline  # type: ignore
except ImportError:
    from libs.figure_factory import Config, Utils, Polyline

cfg = Config(colors=["#000", "#fff", "#f00", "#ccc", "#0f0", "#1f1f1f"])

print(cfg.cnv_props)

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

WIN_NAME = "Window"
CX = cfg.width / 2
CY = cfg.height / 2
BG_COLOR = Utils.hex_to_rgba(cfg.color_palette[5])
# print(points)
# triangle = Polyline(cnv, [point["coords"] for point in points])


# def mouse_callback(event, x, y, _flags, _params):
#     """Mouse callback"""

#     global MOUSE_DOWN, DOT_COUNTER, dot_coords

#     if event == cv.EVENT_RBUTTONUP:
#         DOT_COUNTER = 0
#         dot_coords = dot_coords[:4]

#     MOUSE_DOWN = (
#         True
#         if event == cv.EVENT_LBUTTONDOWN
#         else (False if event == cv.EVENT_LBUTTONUP else MOUSE_DOWN)
#     )

#     if event == cv.EVENT_MOUSEMOVE and MOUSE_DOWN:
#         dot_coords.append([x, y])


def animation() -> None:
    """Main animation"""

    cnv.fill(000)
    cnv[:] = BG_COLOR

    cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
    # cv.setMouseCallback(WIN_NAME, mouse_callback)
    cv.imshow(WIN_NAME, cnv)

    if cv.waitKey(1) & 0xFF == ord("q"):
        return False

    return True


print("Press 'q' for stop")

Utils.animate(animation)

print("Press any key for exit")

cv.waitKey(0)
cv.destroyAllWindows()
