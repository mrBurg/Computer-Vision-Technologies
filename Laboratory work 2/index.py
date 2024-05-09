"""Laboratory work 2"""

import sys
import pathlib
import time
import numpy as np
import cv2 as cv

LIBS_PATH = pathlib.Path(__file__).parent.joinpath("../").resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory import Config, Polyline  # type: ignore
except ImportError:
    from libs.figure_factory import Config, Polyline

cfg = Config()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

CX = cfg.cnv_props[1] / 2

polyline = Polyline(cnv, [[0, 0], [-20, 100], [20, 100]], CX, 600).draw(
    fill_color="#ff00ff"
)

print(cfg.cnv_props)
# while True:
#     time.sleep(1)
#     print("Time")


# cfg = Config(800, 800)
# utils = Utils()

# cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

# CX = cfg.cnv_props[1] / 2

# oval = Oval(300, 200, CX, 130, quality=1)
# rect = Rectangle(400, 200, CX, 300)

# for fig in [oval, rect]:
#     for i in range(5):
#         fig.scale(0.8, 0.8).draw(
#             cnv,
#             stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
#             stroke_width=2,
#         )

# polyline = Polyline([[0, 0], [-20, 100], [20, 100]], CX, 600).scale(4, 1.5)

# for i in range(4):
#     polyline.draw(
#         cnv,
#         fill_color=cfg.color_palette[(i + 1) % len(cfg.color_palette)],
#         stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
#         stroke_width=5,
#     ).rotate(90)

# Oval(50, 50, CX, 600).draw(
#     cnv,
#     stroke_color="#000000",
#     stroke_width=5,
#     fill_color="#ffffff",
# )

cv.imshow("Common Canvas", cnv)  # pylint: disable-msg=E1101
cv.waitKey(0)  # pylint: disable-msg=E1101
cv.destroyAllWindows()  # pylint: disable-msg=E1101
