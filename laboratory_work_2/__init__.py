"""Laboratory work 2"""

# pylint: disable=E1101, W0603

import sys
from pathlib import Path
import time
from random import random, choice
import numpy as np
import cv2 as cv


LIBS_PATH = Path.cwd().resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory import Config, Utils, Line, PolyOval, Rectangle, Polyline  # type: ignore
except ImportError:
    from libs.figure_factory import Config, Utils, Line, PolyOval, Rectangle, Polyline

cfg = Config(600, 600, colors=["#000", "#fff", "#f00", "#ccc", "#0f0", "#1f1f1f"])

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

WIN_NAME = "Window"
CX = cfg.width / 2
CY = cfg.height / 2
BG_COLOR = Utils.hex_to_rgba(cfg.color_palette[5])
COUNTER = 0


def get_random_point_props() -> dict[str, tuple[float, float] | int]:
    """Get random point props"""

    return {
        "coords": [cfg.width * random(), cfg.height * random()],
        "steps": [1 + random() * 4, 1 + random() * 4],
        "direction": choice([1, -1]),
    }


points = [get_random_point_props() for _ in range(3)]
triangle = Polyline(cnv, [point["coords"] for point in points])


def draw_triangle() -> None:
    """Draw triangle"""

    for point in points:
        coords = point["coords"]

        if coords[0] < 0:
            coords[0] = 0

        if coords[0] > cfg.width:
            coords[0] = cfg.width

        if coords[1] < 0:
            coords[1] = 0

        if coords[1] > cfg.height:
            coords[1] = cfg.height

        if (
            coords[0] == 0
            or coords[0] == cfg.width
            or coords[1] == 0
            or coords[1] == cfg.height
        ):
            point["direction"] *= -1
            point["steps"][0] = 1 + random() * 4
            point["steps"][1] = 1 + random() * 4

        coords[0] += point["steps"][0] * point["direction"]
        coords[1] += point["steps"][1] * point["direction"]

    triangle.morph([point["coords"] for point in points]).draw(fill_color="#ff0")


LINE_WIDTH = cfg.width / 60
left_line = Rectangle(cnv, LINE_WIDTH, cfg.height * 3, 0, CY)
right_line = Rectangle(cnv, LINE_WIDTH, cfg.height * 3, cfg.width, CY)
top_line = Rectangle(cnv, cfg.width * 3, LINE_WIDTH, 0, CX)
bottom_line = Rectangle(cnv, cfg.width * 3, LINE_WIDTH, cfg.height, CX)


def draw_lines(sec: int, counter: float) -> None:
    """Draw lines"""

    r, g, b = BG_COLOR
    coef = abs((sec - 30) / 30)
    line_color = (r - round(r * coef), g + round((255 - g) * coef), b - round(b * coef))
    axis_x = LINE_WIDTH * counter % cfg.width
    axis_y = LINE_WIDTH * counter % cfg.height
    rotate = (sec - 30) / 10

    left_line.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(axis_x, CY).rotate(
        rotate
    )
    right_line.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(
        cfg.width - axis_x, CY
    ).rotate(rotate)
    top_line.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(CX, axis_y).rotate(
        rotate
    )
    bottom_line.draw(fill_color=Utils.rgba_to_hex(*line_color)).move(
        CX, cfg.height - axis_y
    ).rotate(rotate)


arrow = Line(cnv, [CX, CY], [CX, CY], 4, cfg.color_palette[4])
oval = PolyOval(cnv, 30, 30, CX, CY)


def draw_clock(current_time: time.struct_time) -> None:
    """Draw clock"""

    def get_angle(data, part, size, bias=0) -> float:
        return [
            np.sin(np.pi / part * data + bias) * size + CX,
            -np.cos(np.pi / part * data + bias) * size + CY,
        ]

    # Seconds
    arrow.draw(
        [CX, CY],
        get_angle(current_time.tm_sec, 30, cfg.width / 2.5),
        stroke_width=4,
        stroke_color=cfg.color_palette[2],
    )

    # Hours
    arrow.draw(
        [CX, CY],
        get_angle(
            current_time.tm_hour,
            6,
            cfg.width / 6,
            np.pi / 360 * current_time.tm_min + np.pi / 21600 * current_time.tm_sec,
        ),
        stroke_width=20,
        stroke_color=cfg.color_palette[1],
    )

    # Minutes
    arrow.draw(
        [CX, CY],
        get_angle(
            current_time.tm_min,
            30,
            cfg.width / 3,
            np.pi / 1800 * current_time.tm_sec,
        ),
        stroke_width=10,
        stroke_color=cfg.color_palette[3],
    )

    oval.draw(
        stroke_width=5,
        stroke_color=cfg.color_palette[0],
        fill_color=cfg.color_palette[1],
    )


dot = PolyOval(cnv, 10, 10, cfg.width / 2, cfg.height / 2)
DOT_COUNTER = 0
SPEED = 5
MOUSE_DOWN = False

dot_coords = [
    [cfg.width / 2, 0],
    [0, cfg.height / 2],
    [cfg.width / 2, cfg.height],
    [cfg.width, cfg.height / 2],
]

dx = dot_coords[0][0] - dot.x
dy = dot_coords[0][1] - dot.y

angle = np.arctan2(dy, dx)
distance = np.sqrt(dx**2 + dy**2)


def mouse_callback(event, x, y, _flags, _params):
    """Mouse callback"""

    global MOUSE_DOWN, DOT_COUNTER, dot_coords

    if event == cv.EVENT_RBUTTONUP:
        DOT_COUNTER = 0
        dot_coords = dot_coords[:4]

    MOUSE_DOWN = (
        True
        if event == cv.EVENT_LBUTTONDOWN
        else (False if event == cv.EVENT_LBUTTONUP else MOUSE_DOWN)
    )

    if event == cv.EVENT_MOUSEMOVE and MOUSE_DOWN:
        dot_coords.append([x, y])


def draw_dot() -> None:
    """Draw red dot"""

    global angle, distance, DOT_COUNTER, dx, dy, dot_coords

    dx = dot_coords[DOT_COUNTER][0] - dot.x
    dy = dot_coords[DOT_COUNTER][1] - dot.y

    angle = np.arctan2(dy, dx)
    distance = np.sqrt(dx**2 + dy**2)

    if distance < SPEED:
        if DOT_COUNTER == len(dot_coords) - 1:
            dot_coords = dot_coords[:4]

        DOT_COUNTER += 1
        DOT_COUNTER = DOT_COUNTER % len(dot_coords)

    dot.translate(np.cos(angle) * SPEED, np.sin(angle) * SPEED).draw(
        fill_color=cfg.color_palette[2]
    )


def animation() -> None:
    """Main animation"""

    global COUNTER

    cnv.fill(000)
    cnv[:] = BG_COLOR

    current_time = time.localtime(time.time())
    # print(f"{current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}")

    draw_triangle()
    draw_lines(current_time.tm_sec, COUNTER)
    draw_clock(current_time)
    draw_dot()

    COUNTER += 1

    cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WIN_NAME, mouse_callback)
    cv.imshow(WIN_NAME, cnv)

    if cv.waitKey(1) & 0xFF == ord("q"):
        return False

    return True


print("Press 'q' for stop")

Utils.animate(animation)

print("Press any key for exit")

cv.waitKey(0)
cv.destroyAllWindows()
