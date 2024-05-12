"""Laboratory work 2"""

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
line_width = cfg.cnv_props[1] / 60
colors = "#000", "#fff", "#f00", "#000", "#ccc", "#0f0"

line = Line(cnv, [CX, CY], [CX, CY], 4, cfg.color_palette[4])
oval = Oval(cnv, 30, 30, CX, CY, quality=1)
rect_l = Rectangle(cnv, line_width, cfg.cnv_props[0] * 2, 0, CY)
rect_r = Rectangle(cnv, line_width, cfg.cnv_props[0] * 2, cfg.cnv_props[1], CY)
rect_t = Rectangle(cnv, cfg.cnv_props[1] * 2, line_width, 0, CX)
rect_b = Rectangle(cnv, cfg.cnv_props[1] * 2, line_width, cfg.cnv_props[0], CX)


def get_random_point_props() -> dict[str, tuple[float, float] | int]:
    """Get random point props"""

    return {
        "coords": [cfg.cnv_props[1] * random(), cfg.cnv_props[0] * random()],
        "steps": [random() * 1, random() * 1],
        "direction": choice([1, -1]),
    }


points = [get_random_point_props() for _ in range(3)]
PADDING = 20

p_line = Polyline(cnv, [point["coords"] for point in points])


def draw_triangle() -> None:
    """Draw triangle"""

    new_points = []

    for point in points:
        coords = point["coords"]

        if (
            coords[0] <= PADDING
            or coords[0] >= cfg.cnv_props[1] - PADDING
            or coords[1] <= PADDING
            or coords[1] >= cfg.cnv_props[0] - PADDING
        ):
            point["direction"] *= -1
            point["steps"][0] = 1 + random() * 5
            point["steps"][1] = 1 + random() * 5

        coords[0] += point["steps"][0] * point["direction"]
        coords[1] += point["steps"][1] * point["direction"]

        new_points.append(point)

    p_line.morph([point["coords"] for point in points]).draw(fill_color="#ff0")


def animation() -> None:
    """Main animation"""

    cnv.fill(255)
    # cnv[:] = utils.hex_to_rgb("#1f1f1f")

    current_time = time.localtime(time.time())

    # print(f"{current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}")

    line_color = (int(195 + current_time.tm_sec), 255, int(195 + current_time.tm_sec))

    ste = line_width * (current_time.tm_sec + 1)
    ets = line_width * (60 - current_time.tm_sec - 1)

    rect_l.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(ste, CY).rotate(1)
    rect_r.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(ets, CY).rotate(1)
    rect_t.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(CX, ste).rotate(1)
    rect_b.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(CX, ets).rotate(1)

    draw_triangle()

    # Seconds
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 30 * (current_time.tm_sec + 1) + Utils.deg_to_rads(INCLINE))
            * SEC_SIZE
            + CX,
            np.cos(-np.pi / 30 * (current_time.tm_sec + 1) + Utils.deg_to_rads(INCLINE))
            * SEC_SIZE
            + CY,
        ],
        stroke_width=4,
        stroke_color=colors[2],
    )

    # Hours
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 6 * (current_time.tm_hour + 1) + Utils.deg_to_rads(INCLINE))
            * HOUR_SIZE
            + CX,
            np.cos(-np.pi / 6 * (current_time.tm_hour + 1) + Utils.deg_to_rads(INCLINE))
            * HOUR_SIZE
            + CY,
        ],
        stroke_width=20,
        stroke_color=colors[3],
    )

    # Minutes
    line.draw(
        [CX, CY],
        [
            np.sin(-np.pi / 30 * (current_time.tm_min + 1) + Utils.deg_to_rads(180))
            * MIN_SIZE
            + CX,
            np.cos(-np.pi / 30 * (current_time.tm_min + 1) + Utils.deg_to_rads(180))
            * MIN_SIZE
            + CY,
        ],
        stroke_width=10,
        stroke_color=colors[4],
    )

    oval.draw(stroke_width=5, stroke_color=colors[0], fill_color=colors[1])

    # cv.namedWindow("Window")
    cv.imshow("Animation 'q' for stop", cnv)  # pylint: disable=E1101

    if cv.waitKey(1) & 0xFF == ord("q"):  # pylint: disable=E1101
        return False

    return True


print("Press 'q' for stop")

Utils.animate(animation)

print("Press any key for exit")
cv.waitKey(0)  # pylint: disable=E1101
cv.destroyAllWindows()  # pylint: disable=E1101
