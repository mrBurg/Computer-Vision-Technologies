"""Laboratory work 1"""

import dataclasses
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from figure_factory import Config, Utils, Oval, Rectangle, Polyline

cfg = Config()
utils = Utils()

cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

CX = cfg.cnv_props[1] / 2

oval = Oval(300, 200, CX, 130, quality=1)
rect = Rectangle(400, 200, CX, 300)

for fig in [oval, rect]:
    for i in range(5):
        fig.scale(0.8, 0.8).draw(
            cnv,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
            stroke_width=2,
        )

polyline = Polyline([[0, 0], [-20, 100], [20, 100]], CX, 600).scale(4, 1.5)

for i in range(4):
    polyline.draw(
        cnv,
        fill_color=cfg.color_palette[(i + 1) % len(cfg.color_palette)],
        stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
        stroke_width=5,
    ).rotate(90)

Oval(50, 50, CX, 600).draw(
    cnv,
    stroke_color="#000000",
    stroke_width=5,
    fill_color="#ffffff",
)

cv.imshow("Common Canvas", cnv)  # pylint: disable-msg=E1101
cv.waitKey(0)  # pylint: disable-msg=E1101
cv.destroyAllWindows()  # pylint: disable-msg=E1101


@dataclasses.dataclass
class Epure:
    """Class Epure"""

    def show(self, student_nimber, coef):
        """Show scheduler"""

        # From, To, Step: To / Step
        px = np.linspace(0, 10, 100)

        plt.title(f"Епюр тестового сигналу для студента №{student_nimber}")
        plt.subplot(2, 2, 1)
        plt.plot(px, student_nimber * coef * np.sin(px), cfg.color_palette[0])
        plt.subplot(2, 2, 2)
        plt.plot(px, (student_nimber + 3) * coef * np.sin(px), cfg.color_palette[1])
        plt.subplot(2, 2, 3)
        plt.plot(px, student_nimber * coef * np.cos(px), cfg.color_palette[2])
        plt.subplot(2, 2, 4)
        plt.plot(px, student_nimber * coef * np.sin(px), cfg.color_palette[0])
        plt.plot(px, (student_nimber + 3) * coef * np.sin(px), cfg.color_palette[1])
        plt.plot(px, student_nimber * coef * np.cos(px), cfg.color_palette[2])
        plt.show()


epure = Epure()
epure.show(1, 0.01)
