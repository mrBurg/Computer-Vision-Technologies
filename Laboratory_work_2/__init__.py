"""Laboratory work 2"""

# pylint: disable=E1101, W0603

import sys
import pathlib
import time
from random import random, choice
import numpy as np
import cv2 as cv


LIBS_PATH = pathlib.Path(__file__).parent.joinpath("../").resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory import Config, Utils, Line, Oval, Rectangle, Polyline  # type: ignore
except ImportError:
    from libs.figure_factory import Config, Utils, Line, Oval, Rectangle, Polyline

cfg = Config(600)

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

CX = cfg.cnv_props[1] / 2
CY = cfg.cnv_props[0] / 2
SEC_SIZE = cfg.cnv_props[1] / 2.5
HOUR_SIZE = cfg.cnv_props[1] / 6
MIN_SIZE = cfg.cnv_props[1] / 3
INCLINE = 180
LINE_WIDTH = cfg.cnv_props[1] / 60
COLORS = "#000", "#fff", "#f00", "#ccc", "#0f0"
BG_COLOR = Utils.hex_to_rgba("#1f1f1f")
COUNTER = 0

line = Line(cnv, [CX, CY], [CX, CY], 4, cfg.color_palette[4])
oval = Oval(cnv, 30, 30, CX, CY, quality=1)
rect_l = Rectangle(cnv, LINE_WIDTH, cfg.cnv_props[0] * 2, 0, CY)
rect_r = Rectangle(cnv, LINE_WIDTH, cfg.cnv_props[0] * 2, cfg.cnv_props[1], CY)
rect_t = Rectangle(cnv, cfg.cnv_props[1] * 2, LINE_WIDTH, 0, CX)
rect_b = Rectangle(cnv, cfg.cnv_props[1] * 2, LINE_WIDTH, cfg.cnv_props[0], CX)


def get_random_point_props() -> dict[str, tuple[float, float] | int]:
    """Get random point props"""

    return {
        "coords": [cfg.cnv_props[1] * random(), cfg.cnv_props[0] * random()],
        "steps": [random() * 5, random() * 5],
        "direction": choice([1, -1]),
    }


points = [get_random_point_props() for _ in range(3)]

p_line = Polyline(cnv, [point["coords"] for point in points])


def draw_triangle() -> None:
    """Draw triangle"""

    for point in points:
        coords = point["coords"]

        if coords[0] <= 0:
            coords[0] = 0

        if coords[1] <= 0:
            coords[1] = 0

        if coords[0] >= cfg.cnv_props[1]:
            coords[0] = cfg.cnv_props[1]

        if coords[1] >= cfg.cnv_props[0]:
            coords[1] = cfg.cnv_props[0]

        if (
            coords[0] == 0
            or coords[0] == cfg.cnv_props[1]
            or coords[1] == 0
            or coords[1] == cfg.cnv_props[0]
        ):
            point["direction"] *= -1
            point["steps"][0] = 2 + random() * 5
            point["steps"][1] = 2 + random() * 5

        coords[0] += point["steps"][0] * point["direction"]
        coords[1] += point["steps"][1] * point["direction"]

    p_line.morph([point["coords"] for point in points]).draw(fill_color="#ff0")


def draw_lines(current_time: any, counter: int) -> None:
    """Draw lines"""

    r, g, b = BG_COLOR
    coef = abs((current_time.tm_sec - 30) / 30)
    line_color = (r - round(r * coef), g + round((255 - g) * coef), b - round(b * coef))
    ste = LINE_WIDTH * (counter % 60)
    ets = LINE_WIDTH * (-counter % 60)

    rect_l.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(ste, CY).rotate(
        (current_time.tm_sec - 30) / 10
    )
    rect_r.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(ets, CY).rotate(
        (current_time.tm_sec - 30) / 10
    )
    rect_t.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(CX, ste).rotate(
        (current_time.tm_sec - 30) / 10
    )
    rect_b.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(CX, ets).rotate(
        (current_time.tm_sec - 30) / 10
    )


def draw_clock(current_time: any) -> None:
    """Draw clock"""

    # Seconds
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 30 * current_time.tm_sec + Utils.deg_to_rads(INCLINE))
            * SEC_SIZE
            + CX,
            np.cos(-np.pi / 30 * current_time.tm_sec + Utils.deg_to_rads(INCLINE))
            * SEC_SIZE
            + CY,
        ],
        stroke_width=4,
        stroke_color=COLORS[2],
    )

    # Hours
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 6 * current_time.tm_hour + Utils.deg_to_rads(INCLINE))
            * HOUR_SIZE
            + CX,
            np.cos(-np.pi / 6 * current_time.tm_hour + Utils.deg_to_rads(INCLINE))
            * HOUR_SIZE
            + CY,
        ],
        stroke_width=20,
        stroke_color=COLORS[1],
    )

    # Minutes
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 30 * current_time.tm_min + Utils.deg_to_rads(180))
            * MIN_SIZE
            + CX,
            np.cos(-np.pi / 30 * current_time.tm_min + Utils.deg_to_rads(180))
            * MIN_SIZE
            + CY,
        ],
        stroke_width=10,
        stroke_color=COLORS[3],
    )

    oval.draw(stroke_width=5, stroke_color=COLORS[0], fill_color=COLORS[1])

    # cv.namedWindow("Window")
    cv.imshow("Animation 'q' for stop", cnv)

    if cv.waitKey(1) & 0xFF == ord("q"):
        return False

    return True


def animation() -> None:
    """Main animation"""

    global COUNTER

    cnv.fill(255)
    cnv[:] = BG_COLOR

    current_time = time.localtime(time.time())
    # print(f"{current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}")

    draw_triangle()
    draw_lines(current_time, COUNTER)
    draw_clock(current_time)

    COUNTER += 0.1

    cv.imshow("Animation 'q' for stop", cnv)

    if cv.waitKey(1) & 0xFF == ord("q"):
        return False

    return True


print("Press 'q' for stop")

Utils.animate(animation)

print("Press any key for exit")

cv.waitKey(0)
cv.destroyAllWindows()
