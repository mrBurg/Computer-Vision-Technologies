"""Laboratory work 3"""

# pylint: disable=E1101, C0412

import sys
from pathlib import Path
import numpy as np
import cv2 as cv


LIBS_PATH = Path.cwd().resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory_3d import Utils, Config, Parallelepiped
except ImportError:
    from libs.figure_factory_3d import Utils, Config, Parallelepiped

try:
    from figure_factory import Line
except ImportError:
    from libs.figure_factory import Line

cfg = Config()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

WIN_NAME = "Window"
CX = cfg.width / 2
CY = cfg.height / 2
BG_COLOR = Utils.hex_to_rgba(cfg.colors[12])

prllppd_config = 200, 100, 50  # width, height, length

# def mouse_callback(event, x, y, _flags, _params):
#     """Mouse callback"""

#     global MOUSE_DOWN, DOT_COUNTER, dot_coords

#     if event == cv.EVENT_RBUTTONUP:q
#         DOT_COUNTER = 0
#         dot_coords = dot_coords[:4]q

#     MOUSE_DOWN = (
#         Trueq
#         if event == cv.EVENT_LBUTTONDOWN
#         else (False if event == cv.EVENT_LBUTTONUP else MOUSE_DOWN)
#     )

#     if event == cv.EVENT_MOUSEMOVE and MOUSE_DOWN:
#         dot_coords.append([x, y])

prllppd = Parallelepiped(
    cnv,
    *prllppd_config,
    offset_x=prllppd_config[0],
    offset_y=prllppd_config[1],
    offset_z=prllppd_config[2],
    stroke_width=2,
    stroke_color=cfg.colors[0],
    # fill_color=cfg.colors[1],
)
prllppd.draw().draw()
# prllppd.translate_3d(250, 0, 0).draw()
# prllppd.translate_3d(200, 200, 0).rotate_x_3d(45).draw()
# print(prllppd)

line = Line(cnv, stroke_color=cfg.colors[6])


def draw_coords() -> None:
    """Draw coords"""

    line.draw([0, CY], [cfg.width, CY]).draw([CX, 0], [CX, cfg.height])


draw_coords()

cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
cv.imshow(WIN_NAME, cnv)


# def draw_parallelepiped() -> None:
#     """Draw parallelepiped"""

#     # rect.draw(stroke_width=5).move(prlpd_config[2], prlpd_config[2])


# def animation() -> None:
#     """Main animation"""

#     cnv.fill(255)
#     cnv[:] = BG_COLOR

#     draw_parallelepiped()
#     draw_coords()

#     cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
#     # cv.setMouseCallback(WIN_NAME, mouse_callback)
#     cv.imshow(WIN_NAME, cnv)

#     if cv.waitKey(1) & 0xFF == ord("q"):
#         return False

#     return True


print("Press 'q' for stop")

# Utils.animate(animation)

print("Press any key for exit")

cv.waitKey(0)
cv.destroyAllWindows()
