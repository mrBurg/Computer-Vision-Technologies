"""Figure Factory"""

# pylint: disable=R0913, R0902, E1101, W0221, W0237, C0301

import dataclasses
import time
from pathlib import Path
from typing import Union, Callable
import numpy as np
import cv2 as cv

__version__ = "3.0.0"

Cell = Union[int, float]

Matrix = tuple[
    tuple[Cell, Cell, Cell],
    tuple[Cell, Cell, Cell],
    tuple[int, int, int],
]
RGBA = tuple[int, int, int, float]
Point = tuple[float, float]
Points = tuple[Point]
Canvas = list[tuple[int, int, int, int | None]]
CanvasProps = tuple[int, tuple[int, int | None] | None]


@dataclasses.dataclass
class Utils:
    """Utils"""

    @staticmethod
    def hex_to_rgba(hex_str: str) -> RGBA:
        """Converts a HEX color to RGB"""

        if hex_str:
            # r = int(hex_str[0:2], 16)
            # g = int(hex_str[2:4], 16)
            # b = int(hex_str[4:6], 16)
            # return (r, g, b)

            rgba = []
            hex_str = hex_str.lstrip("#")

            if len(hex_str) in (3, 4):
                return Utils.hex_to_rgba(f"#{''.join([hex * 2 for hex in hex_str])}")

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

    @staticmethod
    def rgba_to_hex(r: int, g: int, b: int, a: int = 255):
        """Converts a RGB color to HEX"""

        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"  # "#%02x%02x%02x" % (r, g, b)

    @staticmethod
    def deg_to_rads(deg: float) -> float:
        """Converts degrees to radians"""

        return np.radians(deg)  # deg * np.pi / 180

    @staticmethod
    def rads_to_deg(rad: float) -> float:
        """Converts radians to degrees"""

        return np.degrees(rad)  # rad * 180 / np.pi

    @staticmethod
    def animate(animation: Callable[[], None], speed: float = 0.01) -> float:
        """Makes animation"""

        while animation():
            time.sleep(speed)

    @staticmethod
    def description(name: str, props: list = None, args: list = None) -> float:
        """Makes descripton for object"""

        description = f"\n{name} ->"

        if props and len(props):
            description += "\n props:\n  " + "\n  ".join(props)

        if args and len(args):
            description += "\n args:\n  " + "\n  ".join(args)

        return description + "\n"

    @staticmethod
    def get_path(path: str, flag: str = None) -> str:
        """Get path"""

        # Needs TODO: relative r, absolute a, parent p

        files = Path.cwd().glob(f"**/{path}")

        if flag == "All":
            return list(files)

        return list(files)[0]


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

    default_cnv_props = 768, 1024, 3
    cnv_props = default_cnv_props

    default_colors = [
        "#f00",  # 0 rgb("255 0 0") - red (Красный)
        "#ff7400",  # 1 rgb("255 116 0")
        "#ffaa00",  # 2 rgb("255 170 0")
        "#ffd300",  # 3 rgb("255 211 0")
        "#ff0",  # 4 rgb("255 255 0") - yellow (Желтый)
        "#9fee00",  # 5 rgb("159 238 0")
        "#00cc00",  # 6 rgb("0 204 0") - green (Зеленый)
        "#009999",  # 7 rgb("0 153 153") - aqua (Голубой)
        "#1240ab",  # 8 rgb("18 64 171") - blue (Синий)
        "#3914b0",  # 9 rgb("57 20 176")
        "#7109ab",  # 10 rgb("113 9 171")
        "#cd0074",  # 11 rgb("205 0 116") - fuchsia (Розовый)
        "#fff",  # 12 rgb("255 255 255") - white (Белый)
        "#000",  # 13 rgb("0 0 0") - black (Черный)
        "#808080",  # 14 rgb("128 128 128") - gray (Серый)
        "#C0C0C0",  # 15 rgb("192 192 192") - silver (Светло-серый)
    ]
    color_palette = default_colors

    def __init__(
        self,
        *props: CanvasProps,
        depth: int = None,
        colors: list[str] = None,
        add_colors: list[str] = None,
    ) -> None:
        self.width = self.default_cnv_props[1]
        self.height = self.default_cnv_props[0]
        self.depth = max(1, min(depth if depth else self.default_cnv_props[2], 4))

        if colors:
            self.color_palette = colors

        if add_colors:
            self.color_palette.extend(add_colors)

        if len(props) == 1:
            self.width = self.height = props[0]
            self.cnv_props = (self.height, self.width, self.depth)

        elif len(props) == 2:
            self.width = props[0]
            self.height = props[1]
            self.cnv_props = (self.height, self.width, self.depth)

        elif len(props) == 3:
            self.width = props[0]
            self.height = props[1]
            self.depth = max(1, min(depth if depth else props[2], 4))
            self.cnv_props = (self.height, self.width, self.depth)
        else:
            self.cnv_props = self.default_cnv_props

    def __repr__(self):
        return Utils.description(
            self.__class__.__name__,
            [
                f"{prop}: {str(getattr(self, prop, None))}"
                for prop in [
                    "default_cnv_props",
                    "cnv_props",
                    "default_colors",
                    "color_palette",
                ]
            ],
            [
                f"{prop}: {str(getattr(self, prop, None))}"
                for prop in ["width", "height", "depth"]
            ],
        )

    def __str__(self) -> str:
        return repr(self)


@dataclasses.dataclass
class Figure:
    """Figure"""

    x: float = 0
    y: float = 0
    stroke_width: float = 0
    stroke_color: str = None
    fill_color: str = None
    initial_props: Points = None

    def __init__(
        self, cnv: Canvas, points: Points, offset_x: float = x, offset_y: float = y
    ) -> None:
        self.cnv = cnv
        self.initial_props = self.points = points
        self.translate(offset_x, offset_y)

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
    def get_rotate_matrix(rad: float) -> Matrix:
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

        self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))

        sh = sw if sh is None else sh
        self._apply_matrix(Figure.get_scale_matrix(sw, sh))

        self._apply_matrix(Figure.get_translate_matrix(self.x, self.y))

        return self

    def rotate(self, deg: float) -> "Figure":
        """Rotate"""

        self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))
        self._apply_matrix(Figure.get_rotate_matrix(Utils.deg_to_rads(deg)))
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

    def morph(self, points: Points) -> "Figure":
        """Morph"""

        self.points = points

        return self

    def _get_color(self, color: str | bool | None) -> tuple[int, int, int] | None:
        """Get color"""

        if isinstance(color, bool):
            return Utils.hex_to_rgba(self.fill_color) if self.fill_color else None

        self.fill_color = color

        if self.fill_color:
            return Utils.hex_to_rgba(self.fill_color)

        return self.fill_color

    def _get_weight(self, weight: int | float | bool | None) -> int | None:
        """Get weight"""

        if isinstance(weight, bool):
            return int(round(self.stroke_width)) if self.stroke_width else None

        self.stroke_width = weight

        if self.stroke_width:
            return int(round(self.stroke_width))

        return self.stroke_width

    def draw(
        self,
        matrix: Matrix = None,
        stroke_width: int = False,
        stroke_color: str = False,
        fill_color: str = False,
    ) -> "Figure":
        """Draw figure"""

        if matrix:
            self._apply_matrix(self.get_translate_matrix(-self.x, -self.y))
            self._apply_matrix(matrix)
            self._apply_matrix(self.get_translate_matrix(self.x, self.y))

        fill_color = self._get_color(fill_color)
        stroke_width = self._get_weight(stroke_width)
        stroke_color = self._get_color(stroke_color)

        if isinstance(self, Line):
            points = np.array([self.points[0], self.points[1]], dtype=np.int32)

            cv.line(self.cnv, points[0], points[1], stroke_color, stroke_width)
        else:
            points = [np.array(self.points, dtype=np.int32)]

            cv.polylines(
                self.cnv,
                points,
                True,
                stroke_color,
                (
                    int(round(stroke_width * 2))
                    if fill_color and stroke_width
                    else stroke_width
                ),
            )

            if fill_color:
                cv.fillPoly(self.cnv, points, fill_color)

        return self

    def reset(self) -> "Figure":
        """Reset"""

        # Needs TODO: Reset the size when given new

        self.x = 0
        self.y = 0
        self.stroke_width = 0
        self.stroke_color = None
        self.fill_color = None
        self.points = self.initial_props

        return self


@dataclasses.dataclass
class PolyOval(Figure):
    """PolyOval"""

    offset_x: int = 0
    offset_y: int = 0
    start: int = 0
    end: int = 360
    quality: float = 0.75

    def __init__(
        self,
        cnv: Canvas,
        width: int,
        height: int,
        offset_x: int = offset_x,
        offset_y: int = offset_y,
        start: int = start,
        end: int = end,
        quality: float = quality,
    ) -> None:
        self.cnv = cnv
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.start = start
        self.end = end
        self.quality = quality
        self.points = [
            (
                np.sin(Utils.deg_to_rads(degree)) * self.width / 2,
                np.cos(Utils.deg_to_rads(degree)) * self.height / 2,
            )
            for degree in range(
                self.start, self.end, round(20 * (1 - self.quality) + self.quality)
            )
        ]
        super().__init__(self.cnv, self.points, self.offset_x, self.offset_y)

    def __repr__(self):
        return Utils.description(
            self.__class__.__name__,
            [
                f"{prop}: {str(getattr(self, prop, None))}"
                for prop in ["offset_x", "offset_y", "start", "end", "quality"]
            ],
            [
                "cnv: Canvas",
                *(
                    f"{prop}: {str(getattr(self, prop, None))}"
                    for prop in [
                        "width",
                        "height",
                        "offset_x",
                        "offset_y",
                        "start",
                        "end",
                        "quality",
                    ]
                ),
            ],
        )

    def __str__(self) -> str:
        return repr(self)


@dataclasses.dataclass
class Oval(PolyOval):
    """Oval"""

    # Needs TODO: change to cv2.ellipse(cnv, (200, 200), (100, 50), 45, 0, 90, (255, 0, 0), 1)

    def __init__(
        self,
        cnv: Canvas,
        width: int,
        height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        quality: float = 0.75,
    ) -> None:
        self.cnv = cnv
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.quality = quality

        super().__init__(
            self.cnv,
            self.width,
            self.height,
            self.offset_x,
            self.offset_y,
            quality=self.quality,
        )


@dataclasses.dataclass
class Rectangle(Figure):
    """Rectangle"""

    def __init__(
        self, cnv: Canvas, width, height, offset_x: int = 0, offset_y: int = 0
    ) -> None:
        self.cnv = cnv
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y

        p1 = (self.width / -2, self.height / -2)
        p2 = (p1[0] + self.width, p1[1])
        p3 = (p2[0], p1[1] + self.height)
        p4 = (p1[0], p3[1])

        self.points = [p1, p2, p3, p4]

        super().__init__(self.cnv, self.points, self.offset_x, self.offset_y)


@dataclasses.dataclass
class Polyline(Figure):
    """Polyline"""

    def __init__(
        self, cnv: Canvas, points: Points, offset_x: int = 0, offset_y: int = 0
    ) -> None:
        self.cnv = cnv
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.points = points
        super().__init__(self.cnv, self.points, self.offset_x, self.offset_y)


@dataclasses.dataclass
class Line(Figure):
    """Line"""

    point = [0, 0]

    def __init__(
        self,
        cnv: Canvas,
        point: Point = None,
        stroke_width: int = 0,
        stroke_color: str = None,
    ) -> None:
        self.cnv = cnv
        self.line = self.points = [self.point if point is None else point]
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color

        super().__init__(self.cnv, self.points)

    def draw(
        self,
        start_point: Point,
        end_point: Point,
        stroke_width: int = False,
        stroke_color: str = False,
    ) -> None:
        """Draw line"""

        self.points = [start_point, end_point]

        super().draw(None, stroke_width, stroke_color)

        return self

    def add(
        self, x: int, y: int, stroke_width: int = 0, stroke_color: str = None
    ) -> None:
        """Add line"""

        self.stroke_width = stroke_width
        self.stroke_color = stroke_color

        p1 = self.line[-1]
        p2 = [p1[0] + x, p1[1] + y]

        self.line.append(p2)

        self.points = [p1, p2]

        super().draw(None, self.stroke_width, self.stroke_color)

        return self


def test():
    """Test function"""

    cfg = Config(768, 1024)

    print("-" * 10)
    print(cfg)
    print(f"45 degrees in radians is equal to: {Utils.deg_to_rads(45)}")
    print(
        f"{Utils.deg_to_rads(45)} radians in degrees is equal to: {Utils.rads_to_deg(Utils.deg_to_rads(45))}"
    )
    print(
        f"Converting HEX #ff0000[ff] (#f00[f]) to RGB is equal to: {Utils.hex_to_rgba(cfg.color_palette[11])}"
    )
    print("-" * 10)

    width = cfg.width / 4
    height = width
    axis_offset = 150
    rotation_angle = 45

    cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

    oval = Oval(cnv, width, height, axis_offset, axis_offset)
    rect = Rectangle(cnv, width, height, axis_offset, axis_offset)
    polyline = Polyline(
        cnv, [(0, 0.3), (-20, width / 2), (20, width / 2)], axis_offset, axis_offset
    )
    line = Line(cnv)

    for i in range(0, cfg.height, 50):
        line.draw([0, i], [cfg.width, i], stroke_color=cfg.color_palette[6]).draw(
            [i, 0], [i, cfg.height]
        )

    for i, fig in enumerate([rect, oval, polyline]):
        # 1
        fig.draw(fill_color=cfg.color_palette[i % len(cfg.color_palette)])
        # 2
        fig.translate(250, 0).draw(
            stroke_width=5,
            stroke_color=cfg.color_palette[(i + 6) % len(cfg.color_palette)],
            fill_color=cfg.color_palette[i % len(cfg.color_palette)],
        )
        # 3
        fig.translate(250, 0).scale(1, 0.5).draw(
            stroke_color=cfg.color_palette[i % len(cfg.color_palette)],
            fill_color=(
                cfg.color_palette[(i + 1) % len(cfg.color_palette)]
                if isinstance(fig, Polyline)
                else None
            ),
        )
        # 4
        fig.translate(-500, 250).rotate(-rotation_angle).draw(
            stroke_color=None, fill_color=None
        )
        # 5
        fig.translate(250, 0).rotate(rotation_angle).scale(0.5, 2).draw(
            stroke_width=None
        )
        # 6
        fig.draw(
            [
                [0.5, -np.sin(Utils.deg_to_rads(rotation_angle)), 250],
                [np.sin(Utils.deg_to_rads(rotation_angle)), 0.1, 0],
                [0, 0, 1],
            ],
            stroke_width=2,
            stroke_color=(cfg.color_palette[(i + 6) % len(cfg.color_palette)]),
            fill_color=(cfg.color_palette[i % len(cfg.color_palette)]),
        )
        # 7
        fig.reset().translate(150, 650).draw()
        # 8
        fig.translate(250, 0).scale(0.5, 1).rotate(rotation_angle * -3).draw(
            stroke_width=2
        )
        # 9
        fig.move(650, 650).draw(
            fill_color=cfg.color_palette[i % len(cfg.color_palette)]
        )
        # 10
        fig.translate(-500, 250).scale(1, 0.5).rotate(-45).draw(
            stroke_color=cfg.color_palette[(i + 6) % len(cfg.color_palette)],
        )

    line = (
        Line(cnv, [axis_offset, axis_offset], 4, cfg.color_palette[4])
        .add(-axis_offset / 2, 0, 2, cfg.color_palette[4])
        .add(0, axis_offset / 4, 4, cfg.color_palette[5])
        .add(axis_offset / 2, axis_offset / 4, 6, cfg.color_palette[6])
    )
    print(line.line)

    cv.imshow("Common Canvas", cnv)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
