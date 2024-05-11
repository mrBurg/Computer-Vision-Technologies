"""Figure Factory"""

import dataclasses
from typing import Union
import numpy as np
import cv2 as cv

__version__ = "2.2.0"

Cell = Union[int, float]

Matrix = tuple[
    tuple[Cell, Cell, Cell],
    tuple[Cell, Cell, Cell],
    tuple[int, int, int],
]
RGBA = tuple[int, int, int, float]
Point = tuple[float, float]
Points = tuple[tuple[*Point, int]]
Canvas = tuple[int, tuple[int, int | None] | None]


@dataclasses.dataclass
class Config:
    """
    IMPORTANT: the first parameter is height

    args: [ height:int, [ width:int, [ depth:int ]]]
        - If passed as a single integer, represents the same value for both width and height, and the depth is determined by the `depth` argument or by default if argument `depth` is missing
        - If passed two integers, represents (width, height), and the depth is determined by the `depth` argument or by default if argument `depth` is missing
        - If passed three integers, represents (width, height, depth), but the depth is determined by the `depth` argument or the value passed if argument `depth` is missing

    depth: int, optional
        - If the value of depth goes beyond 1 to 4 inclusive, it changes to the nearest limit.

    If there are no arguments, the default settings are taken (width=1024, height=768, depth=3)
    """

    default_cnv_props = 1024, 768, 3
    cnv_props = default_cnv_props
    color_palette = ["#f00f", "#ff7400", "#099", "#0c0", "#cd0074"]

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

    def __repr__(self):
        return f"Config:\n\tdefault_cnv_props: {self.default_cnv_props},\n\tcolor_palette: {self.color_palette}"

    def __str__(self) -> str:
        return repr(self)


@dataclasses.dataclass
class Utils:
    """Utils"""

    def hex_to_rgb(self, hex_str: str) -> RGBA:
        """Converts a HEX color to RGB"""

        # r = int(hex_str[0:2], 16)
        # g = int(hex_str[2:4], 16)
        # b = int(hex_str[4:6], 16)
        # return (r, g, b)

        rgba = []
        hex_str = hex_str.lstrip("#")

        if len(hex_str) in (3, 4):
            return self.hex_to_rgb(f"#{''.join([hex * 2 for hex in hex_str])}")

        if len(hex_str) == 6:
            for i in range(0, 5, 2):
                rgba.insert(0, int(hex_str[i : i + 2], 16))
        elif len(hex_str) == 8:
            for i in range(0, 7, 2):
                color = int(hex_str[i : i + 2], 16)

                if i == 6:
                    color = 1 / 255 * color
                    rgba.append(color)

                    break

                rgba.insert(0, color)
        else:
            rgba = [0, 0, 0, 1.0]

        return tuple(rgba)

    def rgb_to_hex(self, r, g, b):
        """Converts a RGB color to HEX"""

        return f"#{r:02x}{g:02x}{b:02x}"  # "#%02x%02x%02x" % (r, g, b)

    def deg_to_rads(self, deg: float) -> float:
        """Converts degrees to radians"""

        return deg * np.pi / 180.0

    def rads_to_deg(self, rad: float) -> float:
        """Converts radians to degrees"""

        return rad * 180 / np.pi


@dataclasses.dataclass
class Figure(Utils):  # pylint: disable=R0902
    """Figure"""

    x: float = 0
    y: float = 0
    stroke_width: int = 0
    fill_color: str = None
    stroke_color: str = None
    initial_props: Points = None

    def __init__(
        self, cnv: Canvas, points: Points, offset_x: float = x, offset_y: float = y
    ) -> None:
        self.cnv = cnv
        self.initial_props = self.points = points
        self.translate(offset_x, offset_y)

    def __repr__(self):
        return f"axis X:{self.x}, axis Y: {self.y}"

    def __str__(self) -> str:
        return repr(self)

    @staticmethod
    def get_translate_matrix(tx: float, ty: float) -> Matrix:
        """Returns translate matrix"""

        return [
            (1, 0, tx),
            (0, 1, ty),
            (0, 0, 1),
        ]

    @staticmethod
    def get_scale_matrix(sw: float, sh: float) -> Matrix:
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

    def _apply_matrix(self, matrix: Matrix) -> None:
        matrix = np.array(matrix).T

        points = np.array([]).reshape(0, 2)

        for i in self.points:
            point = np.dot([*i, 1], matrix)
            points = np.append(points, [point[:2]], axis=0)

        self.points = points

    def translate(self, tx: float, ty: float = None) -> "Figure":
        """Translate"""

        ty = tx if ty is None else ty
        self.x += tx
        self.y += ty
        self._apply_matrix(Figure.get_translate_matrix(tx, ty))

        return self

    def scale(self, sw: float, sh: float = None) -> "Figure":
        """Scale"""

        sh = sw if sh is None else sh

        self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))
        self._apply_matrix(Figure.get_scale_matrix(sw, sh))
        self._apply_matrix(Figure.get_translate_matrix(self.x, self.y))

        return self

    def rotate(self, deg: float) -> "Figure":
        """Rotate"""

        rad = self.deg_to_rads(deg)

        self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))
        self._apply_matrix(Figure.get_rotate_matrix(rad))
        self._apply_matrix(Figure.get_translate_matrix(self.x, self.y))

        return self

    def move(self, mx: float, my: float = None) -> "Figure":
        """Move"""

        self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))

        my = mx if my is None else my
        self.x = mx
        self.y = my
        self._apply_matrix(Figure.get_translate_matrix(self.x, self.y))

        return self

    def draw(  # pylint: disable=R0913
        self,
        matrix: Matrix = None,
        stroke_width: int = False,
        stroke_color: str = False,
        fill_color: str = False,
        cnv: Canvas = None,
    ) -> "Figure":
        """Draw figure"""

        if matrix:
            self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))
            self._apply_matrix(matrix)
            self._apply_matrix(Figure.get_translate_matrix(self.x, self.y))

        if not isinstance(stroke_width, bool):
            self.stroke_width = int(round(stroke_width)) if stroke_width else None

        if not isinstance(stroke_color, bool):
            self.stroke_color = self.hex_to_rgb(stroke_color) if stroke_color else None

        if isinstance(self, Line):
            cv.line(
                cnv if cnv is not None else self.cnv,
                np.array(self.points[0], dtype=np.int32),
                np.array(self.points[1], dtype=np.int32),
                self.stroke_color,
                self.stroke_width,
            )
        else:
            if not isinstance(fill_color, bool):
                self.fill_color = self.hex_to_rgb(fill_color) if fill_color else None

            points = [np.array(self.points, dtype=np.int32)]

            cv.polylines(  # pylint: disable=E1101
                cnv if cnv is not None else self.cnv,
                points,
                True,
                self.stroke_color,
                (
                    self.stroke_width * 2
                    if self.fill_color and self.stroke_width
                    else self.stroke_width
                ),
            )

            if self.fill_color:
                cv.fillPoly(  # pylint: disable=E1101
                    cnv if cnv is not None else self.cnv, points, self.fill_color
                )

        return self

    def reset(self) -> "Figure":
        """Reset"""

        # Needs TODO: Reset the size when given new

        self.x = 0
        self.y = 0
        self.points = self.initial_props
        self.stroke_width = 0
        self.fill_color = None
        self.stroke_color = None

        return self


@dataclasses.dataclass
class Oval(Figure):
    """Oval"""

    # Needs TODO: change to cv2.ellipse(cnv, (200, 200), (100, 50), 45, 0, 90, (255, 0, 0), 1)

    def __init__(  # pylint: disable=R0913
        self,
        cnv: Canvas,
        width: int,
        height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        start: int = 0,
        end: int = 360,
        quality: float = 0.75,
    ) -> None:
        self.start = start
        self.end = end
        self.quality = quality

        points = [
            (
                np.sin(degree * np.pi / 180) * width / 2,
                np.cos(degree * np.pi / 180) * height / 2,
            )
            for degree in range(
                self.start, self.end, round(20 * (1 - self.quality) + self.quality)
            )
        ]
        super().__init__(cnv, points, offset_x, offset_y)


@dataclasses.dataclass
class Rectangle(Figure):
    """Rectangle"""

    def __init__(  # pylint: disable=R0913
        self, cnv: Canvas, width, height, offset_x: int = 0, offset_y: int = 0
    ) -> None:
        p1 = (width / -2, height / -2)
        p2 = (p1[0] + width, p1[1])
        p3 = (p2[0], p1[1] + height)
        p4 = (p1[0], p3[1])

        super().__init__(cnv, [p1, p2, p3, p4], offset_x, offset_y)


@dataclasses.dataclass
class Polyline(Figure):
    """Polyline"""

    def __init__(
        self, cnv: Canvas, points: Points, offset_x: int = 0, offset_y: int = 0
    ) -> None:
        super().__init__(cnv, points, offset_x, offset_y)


@dataclasses.dataclass
class Line(Figure):
    """Line"""

    def __init__(  # pylint: disable=R0913
        self,
        cnv: Canvas,
        point1: Point,
        point2: Point,
        stroke_width: str = 0,
        stroke_color: str = None,
    ) -> None:
        super().__init__(cnv, [point1, point2])
        super().draw(stroke_width=stroke_width, stroke_color=stroke_color, cnv=cnv)

    def draw(  # pylint: disable=R0913, W0221, W0237
        self,
        start_point: Point,
        end_point: Point,
        matrix: Matrix = None,
        stroke_width: int = False,
        stroke_color: str = False,
        cnv: Canvas = None,
    ) -> None:
        """Draw line"""

        self.points = [start_point, end_point]

        super().draw(matrix, stroke_width, stroke_color, cnv=cnv)

        return self

    def add(self, x: int, y: int) -> None:
        """Add line"""

        self.draw(self.points[1], [self.points[1][0] + x, self.points[1][0] + y])


def test():
    """Test function"""

    cfg = Config()
    utils = Utils()

    print("-" * 10)
    print(cfg)
    print(f"45 degrees in radians is equal to: {utils.deg_to_rads(45)}")
    print(
        f"0.7853981633974483 radians in degrees is equal to: {utils.rads_to_deg(0.7853981633974483)}"  # pylint: disable=C0301
    )
    print(
        f"Converting HEX #ff0000ff (#f00f) to RGB is equal to: {utils.hex_to_rgb('#f00f')}"
    )
    print("-" * 10)

    width = cfg.cnv_props[1] / 4
    height = width
    axis_offset = 150
    rotation_angle = 45

    cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

    oval = Oval(cnv, width, height, axis_offset, axis_offset)
    rect = Rectangle(cnv, width, height, axis_offset, axis_offset)
    polyline = Polyline(
        cnv,
        [
            (0, 0),
            (-20, width / 2),
            (20, width / 2),
        ],
        axis_offset,
        axis_offset,
    )

    for i in range(0, cfg.cnv_props[0], 50):
        Line(cnv, [0, i], [cfg.cnv_props[1], i], stroke_color="#0f0")
        Line(cnv, [i, 0], [i, cfg.cnv_props[0]], stroke_color="#0f0")

    for i, fig in enumerate([rect, oval, polyline]):
        fig.draw(
            fill_color=cfg.color_palette[i % len(cfg.color_palette)],
        )
        fig.translate(300, 0).draw(
            stroke_width=4,
            stroke_color=cfg.color_palette[(i + 1) % len(cfg.color_palette)],
            fill_color=(
                cfg.color_palette[i % len(cfg.color_palette)]
                if isinstance(fig, Polyline)
                else "#fff"
            ),
        )
        fig.translate(-300, 250).scale(1, 0.5).draw(
            stroke_width=4,
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
            fill_color=None,
        )
        fig.translate(300, 50).rotate(-rotation_angle).draw(
            stroke_width=4,
            stroke_color=None,
        )
        fig.translate(-300, 150).rotate(rotation_angle).scale(0.5, 2).draw(
            stroke_width=None
        )
        fig.draw(
            [
                [0.5, -np.sin(utils.deg_to_rads(rotation_angle)), 300],
                [np.sin(utils.deg_to_rads(rotation_angle)), 0.1, 50],
                [0, 0, 1],
            ],
            fill_color=(cfg.color_palette[i % len(cfg.color_palette)]),
        )
        fig.reset().translate(150, 900).draw(
            stroke_width=2,
        )
        fig.scale(0.5, 1).rotate(rotation_angle * -2).translate(300, 0).draw(
            stroke_width=None
        )
        fig.move(650, cfg.cnv_props[0] / 2).draw()

    line = Line(cnv, [150, 150], [200, 200], 4, cfg.color_palette[4])
    line.draw([50, 50], [100, 100], stroke_width=4, stroke_color="fff").add(50, 0)

    cv.imshow("Common Canvas", cnv)  # pylint: disable=E1101
    cv.waitKey(0)  # pylint: disable=E1101
    cv.destroyAllWindows()  # pylint: disable=E1101


if __name__ == "__main__":
    test()
