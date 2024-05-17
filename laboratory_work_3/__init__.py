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
    from figure_factory import Config, Utils, Rectangle, Line  # type: ignore
except ImportError:
    from libs.figure_factory import Config, Utils, Rectangle, Line

cfg = Config()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

WIN_NAME = "Window"
CX = cfg.width / 2
CY = cfg.height / 2
BG_COLOR = Utils.hex_to_rgba(cfg.colors[12])

prlpd_config = 400, 300, 200  # width, height, length

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

line = Line(cnv, stroke_color=cfg.colors[6])


def draw_coords() -> None:
    """Draw coords"""

    line.draw([0, CY], [cfg.width, CY]).draw([CX, 0], [CX, cfg.height])


draw_coords()


prllppd_vertex = []

rect = Rectangle(cnv, prlpd_config[0], prlpd_config[1], offset_x=CX, offset_y=CY).draw()

prllppd_vertex.extend(rect.points)

for i in range(2):
    rect = rect.translate(50, -50).skew_x(45).draw()
    prllppd_vertex.extend(rect.points)

# print(prllppd_vertex)

# ANGLE = 15

# rect.rotate(ANGLE).scale(1, 0.2).rotate(-ANGLE / 5).scale(1, 5).draw()

cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
cv.imshow(WIN_NAME, cnv)


def draw_parallelepiped() -> None:
    """Draw parallelepiped"""

    # rect.draw(stroke_width=5).move(prlpd_config[2], prlpd_config[2])


def animation() -> None:
    """Main animation"""

    cnv.fill(255)
    cnv[:] = BG_COLOR

    draw_parallelepiped()
    draw_coords()

    cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
    # cv.setMouseCallback(WIN_NAME, mouse_callback)
    cv.imshow(WIN_NAME, cnv)

    if cv.waitKey(1) & 0xFF == ord("q"):
        return False

    return True


print("Press 'q' for stop")

# Utils.animate(animation)

print("Press any key for exit")

cv.waitKey(0)
cv.destroyAllWindows()
