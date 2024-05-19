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

prllppd_config = 400, 300, 50  # width, height, length

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

prllppd = Parallelepiped(cnv, *prllppd_config, offset_x=CX, offset_y=CY, offset_z=0)
prllppd.draw()

line = Line(cnv, stroke_color=cfg.colors[6])


def draw_coords() -> None:
    """Draw coords"""

    line.draw([0, CY], [cfg.width, CY]).draw([CX, 0], [CX, cfg.height])


draw_coords()


# prllppd_vertex = np.array([]).reshape(0, 2)

# rect = Rectangle(
#     cnv, prlpd_config[0], prlpd_config[1], offset_x=CX, offset_y=CY, stroke_width=4
# )
# rect.draw()
# prllppd_vertex = np.append(prllppd_vertex, rect.points, axis=0)

# rect.translate(prlpd_config[1] * 0.5, prlpd_config[1] * -0.5)
# rect.draw()
# prllppd_vertex = np.append(prllppd_vertex, rect.points, axis=0)

# # print(prllppd_vertex)

# top = np.array([], DType).reshape(0, 2)
# bottom = np.array([], DType).reshape(0, 2)
# left = np.array([], DType).reshape(0, 2)
# right = np.array([], DType).reshape(0, 2)

# for i in range(0, len(prllppd_vertex), 4):
#     vertices = prllppd_vertex[i : i + 4]

#     top = np.append(top, vertices[0:2], axis=0)
#     bottom = np.append(bottom, vertices[2:4], axis=0)
#     left = np.append(left, [vertices[0], vertices[3]], axis=0)
#     right = np.append(right, vertices[1:3], axis=0)

# # top[-1], top[-2] = top[-2], top[-1].copy()
# # bottom[-1], bottom[-2] = bottom[-2], bottom[-1].copy()
# # left[-1], left[-2] = left[-2], left[-1].copy()
# # right[-1], right[-2] = right[-2], right[-1].copy()

# top = np.concatenate((top[:2], top[-1:], top[-2:-1]))
# bottom = np.concatenate((bottom[:2], bottom[-1:], bottom[-2:-1]))
# left = np.concatenate((left[:2], left[-1:], left[-2:-1]))
# right = np.concatenate((right[:2], right[-1:], right[-2:-1]))

# Polyline(cnv, top).draw(stroke_color=cfg.colors[0], stroke_width=4)
# Polyline(cnv, bottom).draw(stroke_color=cfg.colors[1], stroke_width=4)
# Polyline(cnv, left).draw(stroke_color=cfg.colors[2], stroke_width=4)
# Polyline(cnv, right).draw(stroke_color=cfg.colors[3], stroke_width=4)
######################################

# Polyline(cnv, prllppd_vertex).draw()
# cube = Rectangle(cnv, )


# for i in range(2):
#     rect = rect.translate(50, -50).skew_x(45).draw()
#     prllppd_vertex.extend(rect.points)

# print(prllppd_vertex)

# ANGLE = 15

# rect.rotate(ANGLE).scale(1, 0.2).rotate(-ANGLE / 5).scale(1, 5).draw()

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
