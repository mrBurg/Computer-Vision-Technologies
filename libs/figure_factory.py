"""Figure Factory"""

import dataclasses
from typing import Union
import numpy as np
import cv2 as cv

Matrix = tuple[
    tuple[Union[int, float], Union[int, float], Union[int, float]],
    tuple[Union[int, float], Union[int, float], Union[int, float]],
    tuple[int, int, int],
]
RGBA = tuple[int, int, int, float]
Point = tuple[int, int]
Points = list[tuple[*Point, int]]


@dataclasses.dataclass
class Config:
    """
    Config
    args: [[[ width:int ], height:int ], depth:int ]
        - If passed as a single integer, represents the same value for both width and height, and the depth is determined by the `depth` argument or by default if argument `depth` is missing
        - If passed two integers, represents (width, height), and the depth is determined by the `depth` argument or by default if argument `depth` is missing
        - If passed three integers, represents (width, height, depth), but the depth is determined by the `depth` argument or the value passed if argument `depth` is missing
    depth: int, optional
        - If the value of depth goes beyond 1 to 4 inclusive, it changes to the nearest limit.
    """

    default_cnv_props = (1024, 768, 3)
    color_palette = ["#ff0000ff", "#ff7400", "#009999", "#00cc00", "#cd0074"]

    def __init__(self, *props, depth: int = None) -> None:
        if len(props) == 1:
            self.cnv_props = (
                props[0],
                props[0],
                max(1, min(depth if depth else self.default_cnv_props[2], 4)),
            )
        elif len(props) == 2:
            self.cnv_props = (
                props[0],
                props[1],
                max(1, min(depth if depth else self.default_cnv_props[2], 4)),
            )
        elif len(props) == 3:
            self.cnv_props = (
                props[0],
                props[1],
                max(1, min(depth if depth else props[2], 4)),
            )
        else:
            self.cnv_props = self.default_cnv_props

    def __str__(self) -> str:
        return f"default_cnv_props: {self.default_cnv_props},\n\tcolor_palette: {self.color_palette}"  # pylint: disable-msg=C0301


@dataclasses.dataclass
class Utils:
    """Utils"""

    def hex_to_rgb(self, hex_color: str) -> RGBA:
        """Converts a HEX color to RGB"""

        rgba = []
        hex_color = hex_color.replace("#", "")

        if len(hex_color) == 8:
            for i in range(0, 7, 2):
                color = int(hex_color[i : i + 2], 16)

                if i == 6:
                    color = 1 / 255 * color
                    rgba.append(color)

                    break

                rgba.insert(0, color)
        elif len(hex_color) == 6:
            for i in range(0, 5, 2):
                rgba.insert(0, int(hex_color[i : i + 2], 16))
        else:
            rgba = [0, 0, 0, 1.0]

        return tuple(rgba)

    def deg_to_rads(self, deg: float) -> float:
        """Converts degrees to radians"""

        return deg * np.pi / 180.0

    def rads_to_deg(self, rad: float) -> float:
        """Converts radians to degrees"""

        return rad * 180 / np.pi


@dataclasses.dataclass
class Figure(Utils):
    """Figure"""

    axis_x = 0
    axis_y = 0

    def __init__(self, points: Points, offset_x: int = 0, offset_y: int = 0) -> None:
        self.initial_props = self.points = np.array(points, np.int32)
        self.translate(offset_x, offset_y)

    @staticmethod
    def get_translate_matrix(tx: int, ty: int) -> Matrix:
        """Returns translate matrix"""

        return [
            (1, 0, tx),
            (0, 1, ty),
            (0, 0, 1),
        ]

    @staticmethod
    def get_scale_matrix(sw: int, sh: int) -> Matrix:
        """Returns scale matrix"""

        return [
            (sw, 0, 0),
            (0, sh, 0),
            (0, 0, 1),
        ]

    @staticmethod
    def get_rotate_matrix(
        rad: float,
    ) -> Matrix:
        """Returns rotate matrix"""

        return [
            (np.cos(rad), -np.sin(rad), 0),
            (np.sin(rad), np.cos(rad), 0),
            (0, 0, 1),
        ]

    def _apply_matrix(
        self,
        matrix: Matrix,
    ) -> None:
        matrix = np.array(matrix).T

        points = np.array([]).reshape(0, 2)

        for i in self.points:
            point = np.dot([*i, 1], matrix)
            points = np.append(points, [point[:2]], axis=0)

        self.points = np.array(points, dtype=np.int32)

    def translate(self, tx: int, ty: int = None) -> "Figure":
        """Translate"""

        if ty is None:
            ty = tx

        self.axis_x += tx
        self.axis_y += ty
        self._apply_matrix(Figure.get_translate_matrix(tx, ty))

        return self

    def scale(self, sw: int, sh: int = None) -> "Figure":
        """Scale"""

        if sh is None:
            sh = sw

        self._apply_matrix(Figure.get_translate_matrix(-self.axis_x, -self.axis_y))
        self._apply_matrix(Figure.get_scale_matrix(sw, sh))
        self._apply_matrix(Figure.get_translate_matrix(self.axis_x, self.axis_y))

        return self

    def rotate(self, deg: float) -> "Figure":
        """Rotate"""

        rad = self.deg_to_rads(deg)

        self._apply_matrix(Figure.get_translate_matrix(-self.axis_x, -self.axis_y))
        self._apply_matrix(Figure.get_rotate_matrix(rad))
        self._apply_matrix(Figure.get_translate_matrix(self.axis_x, self.axis_y))

        return self

    def draw(  # pylint: disable-msg=R0913
        self,
        cnv,
        matrix: Matrix = None,
        stroke_width: int = None,
        stroke_color: str = None,
        fill_color: str = None,
    ) -> "Figure":
        """Draw figure"""

        if matrix:
            self._apply_matrix(Figure.get_translate_matrix(-self.axis_x, -self.axis_y))
            self._apply_matrix(matrix)
            self._apply_matrix(Figure.get_translate_matrix(self.axis_x, self.axis_y))

        if stroke_width:
            stroke_width = int(round(stroke_width))

        if stroke_color:
            stroke_color = self.hex_to_rgb(stroke_color)

        if fill_color:
            fill_color = self.hex_to_rgb(fill_color)

        cv.polylines(  # pylint: disable-msg=E1101
            cnv,
            [self.points],
            True,
            stroke_color,
            stroke_width * 2 if fill_color else stroke_width,
        )

        if fill_color:
            cv.fillPoly(cnv, [self.points], fill_color)  # pylint: disable-msg=E1101

        return self

    def reset(self) -> "Figure":
        """Reset"""

        # Needs TODO: Reset the size when given new

        self.axis_x = 0
        self.axis_y = 0
        self.points = self.initial_props

        return self


@dataclasses.dataclass
class Oval(Figure):
    """Oval"""

    # Needs TODO: change to cv2.ellipse(cnv, (200, 200), (100, 50), 45, 0, 90, (255, 0, 0), 1)

    def __init__(  # pylint: disable-msg=R0913
        self,
        width: int = 0,
        height: int = 0,
        offset_x: int = 0,
        offset_y: int = 0,
        start: int = 0,
        end: int = 360,
        quality: float = 0.75,
    ) -> None:
        self.start = start
        self.end = end
        self.quality = quality

        points = np.array(
            [
                (
                    np.sin(degree * np.pi / 180) * width / 2,
                    np.cos(degree * np.pi / 180) * height / 2,
                )
                for degree in range(
                    self.start, self.end, round(20 * (1 - self.quality) + self.quality)
                )
            ]
        )
        super().__init__(points, offset_x, offset_y)


@dataclasses.dataclass
class Rectangle(Figure):
    """Rectangle"""

    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        p1 = (width / -2, height / -2)
        p2 = (p1[0] + width, p1[1])
        p3 = (p2[0], p1[1] + height)
        p4 = (p1[0], p3[1])

        super().__init__([p1, p2, p3, p4], offset_x, offset_y)


@dataclasses.dataclass
class Polyline(Figure):
    """Polyline"""

    # Needs fix: Should fix this
    def __init__(  # pylint: disable-msg=W0246
        self, points: Points, offset_x: int = 0, offset_y: int = 0
    ) -> None:
        super().__init__(points, offset_x, offset_y)


@dataclasses.dataclass
class Line(Utils):
    """Line"""

    def draw(  # pylint: disable-msg=R0913
        self,
        cnv,
        start_point: Point,
        end_point: Point,
        stroke_color: str = None,
        stroke_width: int = None,
    ) -> None:
        """Draw line"""

        if stroke_color:
            stroke_color = self.hex_to_rgb(stroke_color)

        if stroke_width:
            stroke_width = int(round(stroke_width))

        cv.line(  # pylint: disable-msg=E1101
            cnv,
            np.array(start_point, np.int32),
            np.array(end_point, np.int32),
            stroke_color,
            stroke_width,
        )


def test():
    """Test function"""

    cfg = Config()
    utils = Utils()

    print("-" * 10)
    print(f"Default config:\n\t{cfg}")
    print(f"45 degrees in radians is equal to: {utils.deg_to_rads(45)}")
    print(
        f"0.7853981633974483 radians in degrees is equal to: {utils.rads_to_deg(0.7853981633974483)}"  # pylint: disable-msg=C0301
    )
    print(
        f"Converting HEX #ff0000ff to RGB is equal to: {utils.hex_to_rgb('#ff0000ff')}"
    )
    print("-" * 10)

    cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

    width = cfg.cnv_props[1] / 4
    height = width
    axis_offset = 150
    rotation_angle = 45

    line = Line()
    rect = Rectangle(width, height, axis_offset, axis_offset)
    oval = Oval(width, height, axis_offset, axis_offset)
    polyline = Polyline(
        [
            (0, 0),
            (-20, width / 2),
            (20, width / 2),
        ],
        axis_offset,
        axis_offset,
    )

    for i in range(0, cfg.cnv_props[0], 50):
        line.draw(cnv, [0, i], [cfg.cnv_props[1], i], stroke_color="#00ff00")
        line.draw(cnv, [i, 0], [i, cfg.cnv_props[0]], stroke_color="#00ff00")

    for i, fig in enumerate([rect, oval, polyline]):
        fig.draw(
            cnv,
            stroke_width=2,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
            fill_color=cfg.color_palette[i % len(cfg.color_palette)],
        )
        fig.translate(300, 0).draw(
            cnv,
            stroke_width=2,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
            fill_color=(
                cfg.color_palette[i % len(cfg.color_palette)]
                if isinstance(fig, Polyline)
                else "#ffffff"
            ),
        )
        fig.translate(-300, 250).scale(1, 0.5).draw(
            cnv,
            stroke_width=2,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
        )
        fig.translate(300, 50).rotate(-rotation_angle).draw(
            cnv,
            stroke_width=2,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
        )
        fig.translate(-300, 150).rotate(rotation_angle).scale(0.5, 2).draw(
            cnv,
            stroke_width=2,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
        )
        fig.draw(
            cnv,
            [
                [0.5, -np.sin(utils.deg_to_rads(rotation_angle)), 300],
                [np.sin(utils.deg_to_rads(rotation_angle)), 0.1, 50],
                [0, 0, 1],
            ],
            stroke_width=2,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
        )
        fig.reset().translate(150, 900).draw(cnv, stroke_width=2)
        fig.scale(0.5, 1).rotate(rotation_angle * -2).translate(300, 0).draw(
            cnv, stroke_width=2
        )

    cv.imshow("Common Canvas", cnv)  # pylint: disable-msg=E1101
    cv.waitKey(0)  # pylint: disable-msg=E1101
    cv.destroyAllWindows()  # pylint: disable-msg=E1101


if __name__ == "__main__":
    test()
