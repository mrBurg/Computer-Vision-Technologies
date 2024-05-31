"""Laboratory work 4"""

# pylint: disable=E1101, C0412, W0603

import sys
from random import random
from pathlib import Path
import numpy as np
import cv2 as cv

LIBS_PATH = Path.cwd().resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory_3d import Utils, Config, Parallelepiped, RGB
except ImportError:
    from libs.figure_factory_3d import Utils, Config, Parallelepiped, RGB

cfg = Config()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

WIN_NAME = "Window"
CX = cfg.width / 2
CY = cfg.height / 2
BG_COLOR = Utils.hex_to_rgba(cfg.colors[12])
STROKE_COLOR = Utils.hex_to_rgba(cfg.colors[0])
FILL_COLOR = Utils.hex_to_rgba(cfg.colors[5])
prllppd_config = 400, 350, 200
RX = RY = TX = TY = COUNTER = 0

prllppd = (
    Parallelepiped(
        cnv,
        *prllppd_config,
        stroke_width=3,
        stroke_color=cfg.colors[4],
        fill_color=cfg.colors[6]
    )
    .translate_3d(CX, CY, 0)
    .rotate_3d(random() * 45, random() * 45, random() * 45)
)


def mouse_callback(event, x, y, _flags, _params):
    """Mouse callback"""

    global RX, RY

    if event == cv.EVENT_MOUSEMOVE:
        RX = 100 / prllppd.x * (x - prllppd.x) / 100
        RY = 100 / prllppd.y * (y - prllppd.y) / -100


def keyboard_callback(key):
    """Keyboard callback"""

    global TX, TY

    speed = 5

    if key == ord("q"):
        return False

    if key == ord("a"):
        TX = -speed

    if key == ord("d"):
        TX = speed

    if key == ord("w"):
        TY = -speed

    if key == ord("s"):
        TY = speed

    if key == ord("c"):
        TX = TY = 0

    return True


def get_residual_color(color: RGB) -> RGB:
    """Get color"""

    return (255 - color[0], 255 - color[1], 255 - color[2])


def get_color(color: RGB, residual_color: RGB, coef: float):
    """Get color"""

    return (
        color[0] + abs(round(residual_color[0] * coef)),
        color[1] + abs(round(residual_color[1] * coef)),
        color[2] + abs(round(residual_color[2] * coef)),
    )


sc = get_residual_color(STROKE_COLOR)
fc = get_residual_color(FILL_COLOR)


def animation() -> None:
    """Main animation"""

    global COUNTER

    cnv.fill(255)
    cnv[:] = BG_COLOR

    s_coef = np.sin(COUNTER / 50)
    f_coef = np.sin(COUNTER / 100)

    cfg.grid(cnv, size=200, position=(int(CX), int(CY)))

    prllppd.translate_3d(TX, TY, 0).rotate_3d(RY, RX, 0).draw(
        stroke_color=Utils.rgba_to_hex(*get_color(STROKE_COLOR, sc, s_coef)),
        fill_color=Utils.rgba_to_hex(*get_color(FILL_COLOR, fc, f_coef)),
    )

    if prllppd.x < 0:
        prllppd.move_3d(cfg.width, prllppd.y, 0)

    if prllppd.x > cfg.width:
        prllppd.move_3d(0, prllppd.y, 0)

    if prllppd.y < 0:
        prllppd.move_3d(prllppd.x, cfg.height, 0)

    if prllppd.y > cfg.height:
        prllppd.move_3d(prllppd.x, 0, 0)

    COUNTER += 1

    cv.imshow(WIN_NAME, cnv)

    key = cv.waitKey(1) & 0xFF

    return keyboard_callback(key)


print("Move the mouse left and right to rotate along the Y axis")
print("Move the mouse up and down to rotate along the X axis")
print("Press the 'A' and 'D' to move along the X axis")
print("Press the 'W' and 'S' to move along the Y axis")
print("Press the 'C' to stop moving")
print("Press 'Q' for stop")

cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
cv.setMouseCallback(WIN_NAME, mouse_callback)
Utils.animate(animation, 0.025)

print("Press any key for exit")

cv.waitKey(0)
cv.destroyAllWindows()
