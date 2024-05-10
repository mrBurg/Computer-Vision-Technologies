"""Laboratory work 2"""

import sys
import pathlib
import time
import numpy as np
import cv2 as cv


LIBS_PATH = pathlib.Path(__file__).parent.joinpath("../").resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory import Config, Utils, Line, Oval, Rectangle  # type: ignore
except ImportError:
    from libs.figure_factory import Config, Utils, Line, Oval, Rectangle

cfg = Config(800)
utils = Utils()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

CX = cfg.cnv_props[0] / 2
CY = cfg.cnv_props[1] / 2
SEC_SIZE = 300
HOUR_SIZE = 100
MIN_SIZE = 200
INCLINE = 180
rect_width = cfg.cnv_props[0] / 60
colors = "#000", "#fff", "#f00", "#000", "#ccc", "#0f0"

current_time = time.localtime(time.time())

line = Line(
    cnv,
    [CX, CY],
    [
        np.sin(np.pi / 30 * current_time.tm_sec) * SEC_SIZE + CX,
        np.cos(np.pi / 30 * current_time.tm_sec) * SEC_SIZE + CX,
    ],
    4,
    cfg.color_palette[4],
)
oval = Oval(cnv, 50, 50, CX, CY, quality=1)
rect = Rectangle(cnv, rect_width, cfg.cnv_props[1], 0, CX)

# cv.namedWindow("Analog clock")  # pylint: disable=E1101
# print(utils.hex_to_rgb("#0f0"))
# print(utils.rgb_to_hex(127, 127, 127))

print("Press 'q' for stop")
while True:
    cnv.fill(255)
    # cnv[:] = utils.hex_to_rgb("#1f1f1f")

    current_time = time.localtime(time.time())

    # print(f"{current_time.tm_hour}:{current_timecfg.cnv_props.tm_min}:{current_time.tm_sec}")

    if rect.axis_x > cfg.cnv_props[0]:
        rect.reset().translate(0, CY)

    rect.draw(fill_color=colors[5]).translate(rect_width, 0)

    # for i in range(60):
    #     if rect.axis_x > cfg.cnv_props[0]:
    #         rect.reset().translate(0, CY)

    #     rect.draw(
    #         fill_color=utils.rgb_to_hex(i * 4, round(255 - i * 4), i * 4)
    #     ).translate(rect_width, 0)

    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 30 * (current_time.tm_sec + 1) + utils.deg_to_rads(INCLINE))
            * SEC_SIZE
            + CX,
            np.cos(-np.pi / 30 * (current_time.tm_sec + 1) + utils.deg_to_rads(INCLINE))
            * SEC_SIZE
            + CX,
        ],
        stroke_width=4,
        stroke_color=colors[2],
    )
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 6 * (current_time.tm_hour + 1) + utils.deg_to_rads(INCLINE))
            * HOUR_SIZE
            + CX,
            np.cos(-np.pi / 6 * (current_time.tm_hour + 1) + utils.deg_to_rads(INCLINE))
            * HOUR_SIZE
            + CX,
        ],
        stroke_width=20,
        stroke_color=colors[3],
    )
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 30 * (current_time.tm_min + 1) + utils.deg_to_rads(180))
            * MIN_SIZE
            + CX,
            np.cos(-np.pi / 30 * (current_time.tm_min + 1) + utils.deg_to_rads(180))
            * MIN_SIZE
            + CX,
        ],
        stroke_width=10,
        stroke_color=colors[4],
    )

    oval.draw(stroke_width=5, stroke_color=colors[0], fill_color=colors[1])

    cv.imshow("Animation 'q' for stop", cnv)  # pylint: disable=E1101

    time.sleep(1)

    if cv.waitKey(1) & 0xFF == ord("q"):  # pylint: disable=E1101
        break

print("Press any key for exit")
cv.waitKey(0)
cv.destroyAllWindows()
