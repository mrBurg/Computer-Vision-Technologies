"""Laboratory work 2"""

import sys
import pathlib
import time
import numpy as np
import cv2 as cv


LIBS_PATH = pathlib.Path(__file__).parent.joinpath("../").resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory import Config, Utils, Line, Oval  # type: ignore
except ImportError:
    from libs.figure_factory import Config, Utils, Line, Oval

cfg = Config(800)
utils = Utils()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

CX = cfg.cnv_props[0] / 2
CY = cfg.cnv_props[1] / 2
SEC_SIZE = 300
HOUR_SIZE = 100
MIN_SIZE = 200
INCLINE = 180

current_time = time.localtime(time.time())

line = Line(
    cnv,
    [CX, CY],
    [
        np.sin(utils.deg_to_rads(-current_time.tm_sec)) * SEC_SIZE + CX,
        np.cos(utils.deg_to_rads(-current_time.tm_sec)) * SEC_SIZE + CX,
    ],
    4,
    cfg.color_palette[4],
)
oval = Oval(cnv, 50, 50, CX, CY, quality=1)
COUNT = 0

# cv.namedWindow("Analog clock")  # pylint: disable=E1101

while True:
    cnv.fill(255)
    # cnv[:] = utils.hex_to_rgb("#1f1f1f")

    current_time = time.localtime(time.time())

    print(f"{current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}")

    line.draw(
        [CX, CY],
        [
            np.sin(utils.deg_to_rads(-current_time.tm_sec * 6 + INCLINE)) * SEC_SIZE
            + CX,
            np.cos(utils.deg_to_rads(-current_time.tm_sec * 6 + INCLINE)) * SEC_SIZE
            + CX,
        ],
        stroke_width=4,
        stroke_color="#f00",
    )
    line.draw(
        [CX, CY],
        [
            np.sin(utils.deg_to_rads((-current_time.tm_hour) * 12 + INCLINE))
            * HOUR_SIZE
            + CX,
            np.cos(utils.deg_to_rads((-current_time.tm_hour) * 12 + INCLINE))
            * HOUR_SIZE
            + CX,
        ],
        stroke_width=20,
        stroke_color="#000",
    )
    line.draw(
        [CX, CY],
        [
            np.sin(utils.deg_to_rads(-current_time.tm_min * 6 + INCLINE)) * MIN_SIZE
            + CX,
            np.cos(utils.deg_to_rads(-current_time.tm_min * 6 + INCLINE)) * MIN_SIZE
            + CX,
        ],
        stroke_width=10,
        stroke_color="#ccc",
    )
    COUNT += 1
    oval.draw(stroke_width=5, stroke_color="#000", fill_color="#fff")

    cv.imshow("Animation 'q' for stop", cnv)  # pylint: disable=E1101

    time.sleep(1)

    if cv.waitKey(1) & 0xFF == ord("q"):  # pylint: disable=E1101
        break

cv.waitKey(0)
cv.destroyAllWindows()
