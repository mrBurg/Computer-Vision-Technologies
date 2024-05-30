"""Laboratory work 3"""

# pylint: disable=E1101, C0412, W0603

import sys
from random import random
from pathlib import Path
import numpy as np
import cv2 as cv

LIBS_PATH = Path.cwd().resolve()

sys.path.append(str(LIBS_PATH))

try:
    from figure_factory_3d import Utils, Config, Parallelepiped
except ImportError:
    from libs.figure_factory_3d import Utils, Config, Parallelepiped

cfg = Config()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

WIN_NAME = "Window"
CX = cfg.width / 2
CY = cfg.height / 2
BG_COLOR = Utils.hex_to_rgba(cfg.colors[12])
STROKE_COLOR = Utils.hex_to_rgba(cfg.colors[5])
prllppd_config = 400, 350, 200
RX = RY = TX = TY = 0
COUNTER = 0

prllppd = (
    Parallelepiped(cnv, *prllppd_config, stroke_width=5)
    .translate_3d(CX, CY, 0)
    .rotate_3d(random() * 45, random() * 45, random() * 45)
)


def mouse_callback(event, x, y, _flags, _params):
    """Mouse callback"""

    global RX, RY

    if event == cv.EVENT_MOUSEMOVE:
        RX = 100 / prllppd.x * (x - prllppd.x) / -100
        RY = 100 / prllppd.y * (y - prllppd.y) / -100


def keyboard_callback(key):
    """Keyboard callback"""

    global TX, TY

    if key == ord("q"):
        return False

    if key == ord("a"):
        TX = -1

    if key == ord("d"):
        TX = 1

    if key == ord("w"):
        TY = -1

    if key == ord("s"):
        TY = 1

    if key == ord("c"):
        TX = TY = 0

    return True


b = 255 - STROKE_COLOR[0]
g = 255 - STROKE_COLOR[1]
r = 255 - STROKE_COLOR[2]


def animation() -> None:
    """Main animation"""

    global COUNTER

    cnv.fill(255)
    cnv[:] = BG_COLOR

    coef = np.sin(COUNTER / min(b, g, r))

    cfg.grid(cnv, size=200, position=(int(CX), int(CY)))

    prllppd.translate_3d(TX, TY, 0).rotate_3d(RY, RX, 0).draw(
        stroke_color=Utils.rgba_to_hex(
            STROKE_COLOR[0] + abs(round(b * coef)),
            STROKE_COLOR[1] + abs(round(g * coef)),
            STROKE_COLOR[2] + abs(round(r * coef)),
        )
    )

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
