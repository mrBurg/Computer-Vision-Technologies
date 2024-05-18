"""Figure Factory"""

# pylint: disable=E1101, W0237, R0913, R0902, C0301

from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional
import numpy as np
import cv2 as cv
from utils import Utils, RGB
from config import Config

__version__ = "3.0.0"

Cell = Union[int, float]

Matrix = Tuple[
    Tuple[Cell, Cell, Cell],
    Tuple[Cell, Cell, Cell],
    Tuple[int, int, int],
]
Point = Tuple[float, float]
Points = List[Point]
Canvas = List[Tuple[int, int, int, Union[int, None]]]


@dataclass
class Figure:
    """Figure"""

    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    pivot: Point = field(default_factory=lambda: [0, 0])
    stroke_width: Union[float, bool] = False
    stroke_color: Union[str, bool] = False
    fill_color: Union[str, bool] = False
    initial_props: Points = None

    def __init__(
        self,
        cnv: Canvas,
        points: Points,
        width: float = width,
        height: float = height,
        # **kwargs,
        offset_x: float = x,
        offset_y: float = y,
        stroke_width: float = stroke_width,
        stroke_color: str = stroke_color,
        fill_color: str = fill_color,
    ) -> None:
        self.cnv = cnv
        self.width = width
        self.height = height
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.fill_color = fill_color
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

    @staticmethod
    def get_skew_x_matrix(deg: float) -> Matrix:
        """Returns skewX matrix"""

        return [
            (1, np.tan(deg), 0),
            (0, 1, 0),
            (0, 0, 1),
        ]

    @staticmethod
    def get_skew_y_matrix(deg: float) -> Matrix:
        """Returns skewY matrix"""

        return [
            (1, 0, 0),
            (np.tan(deg), 1, 0),
            (0, 0, 1),
        ]

    def _apply_matrix(self, matrix: Matrix) -> None:
        matrix = np.array(matrix).T

        points = np.array([]).reshape(0, 2)

        for i in self.points:
            point = np.dot([*i, 1], matrix)
            points = np.append(points, [point[:2]], axis=0)

        self.points = points

    def _get_stroke_width(
        self, weight: Optional[Union[int, float, bool]]
    ) -> Union[int, None]:
        """Get weight"""

        if isinstance(weight, bool):
            if self.stroke_width:  # pylint: disable=R1705
                return int(round(self.stroke_width))

            elif self.stroke_width is None:
                return None

            if weight:
                self.stroke_width = 1

                return self.stroke_width

            self.stroke_width = 0

            return self.stroke_width

        if isinstance(weight, (int, float)):
            self.stroke_width = int(round(weight))

            return self.stroke_width

        self.stroke_width = None

        return None

    def _get_stroke_color(  # pylint: disable=R0911
        self, color: Optional[Union[str | bool]]
    ) -> RGB | None:
        """Get color"""

        if isinstance(color, bool):
            if self.stroke_color:  # pylint: disable=R1705
                if isinstance(self.stroke_color, bool):
                    return Utils.hex_to_rgba("#fff")

                return Utils.hex_to_rgba(self.stroke_color)

            elif self.stroke_color is None:
                return None

            if color:
                self.stroke_color = "#fff"

                return Utils.hex_to_rgba(self.stroke_color)

            self.stroke_color = "#000"

            return Utils.hex_to_rgba(self.stroke_color)

        if isinstance(color, str):
            self.stroke_color = color

            return Utils.hex_to_rgba(self.stroke_color)

        self.stroke_color = None

        return None

    def _get_fill_color(  # pylint: disable=R0911
        self, color: Optional[Union[str | bool]]
    ) -> RGB | None:
        """Get color"""

        if isinstance(color, bool):
            if self.fill_color:  # pylint: disable=R1705
                if isinstance(self.fill_color, bool):
                    return Utils.hex_to_rgba("#fff")

                return Utils.hex_to_rgba(self.fill_color)

            elif self.fill_color is None:
                return None

            self.fill_color = None

            return self.fill_color

        if isinstance(color, str):
            self.fill_color = color

            return Utils.hex_to_rgba(self.fill_color)

        self.fill_color = None

        return None

    def set_pivot(self, px: float, py: float = None) -> "Figure":
        """Set Pivot"""

        py = px if py is None else py
        self.pivot = [px, py]

        self._apply_matrix(
            Figure.get_translate_matrix(
                self.pivot[0] * self.width - self.width / 2,
                self.pivot[1] * self.height - self.height / 2,
            )
        )

        return self

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

    def skew_x(self, deg: float) -> "Figure":
        """skew X"""

        self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))
        self._apply_matrix(Figure.get_skew_x_matrix(Utils.deg_to_rads(deg)))
        self._apply_matrix(Figure.get_translate_matrix(self.x, self.y))

        return self

    def skew_y(self, deg: float) -> "Figure":
        """skew Y"""

        self._apply_matrix(Figure.get_translate_matrix(-self.x, -self.y))
        self._apply_matrix(Figure.get_skew_y_matrix(Utils.deg_to_rads(deg)))
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

    def draw(
        self,
        matrix: Matrix = None,
        stroke_width: Optional[Union[int, float, bool]] = False,
        stroke_color: Optional[Union[str | bool]] = False,
        fill_color: Optional[Union[str | bool]] = False,
    ) -> "Figure":
        """Draw figure"""

        if matrix:
            self._apply_matrix(self.get_translate_matrix(-self.x, -self.y))
            self._apply_matrix(matrix)
            self._apply_matrix(self.get_translate_matrix(self.x, self.y))

        stroke_width = self._get_stroke_width(stroke_width)
        stroke_color = self._get_stroke_color(stroke_color)
        fill_color = self._get_fill_color(fill_color)

        if isinstance(self, Line):
            points = np.array([self.points[0], self.points[1]], dtype=np.int32)

            cv.line(
                self.cnv,
                points[0],
                points[1],
                stroke_color,
                1 if stroke_width < 1 else stroke_width,
            )
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

        self.x = self.y = 0
        self.stroke_width = 1
        self.stroke_color = self.fill_color = None
        self.points = self.initial_props

        return self


@dataclass
class PolyOval(Figure):
    """PolyOval"""

    start: int = 0
    end: int = 360
    quality: float = 0.75

    def __init__(
        self,
        cnv: Canvas,
        width: float,
        height: float,
        start: float = start,
        end: float = end,
        quality: float = quality,
        **kwargs,
    ) -> None:
        self.start = start
        self.end = end
        self.quality = quality
        self.points = [
            (
                np.sin(Utils.deg_to_rads(degree)) * width / 2,
                np.cos(Utils.deg_to_rads(degree)) * height / 2,
            )
            for degree in range(
                self.start, self.end, round(20 * (1 - self.quality) + self.quality)
            )
        ]

        super().__init__(cnv, self.points, width, height, **kwargs)

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


@dataclass
class Oval(PolyOval):
    """Oval"""

    # Needs TODO: change to cv2.ellipse(cnv, (200, 200), (100, 50), 45, 0, 90, (255, 0, 0), 1)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@dataclass
class Rectangle(Figure):
    """Rectangle"""

    def __init__(self, cnv: Canvas, width, height, **kwargs) -> None:
        self.cnv = cnv

        p1 = (width / -2, height / -2)
        p2 = (p1[0] + width, p1[1])
        p3 = (p2[0], p1[1] + height)
        p4 = (p1[0], p3[1])

        self.points = [p1, p2, p3, p4]

        super().__init__(self.cnv, self.points, width, height, **kwargs)


@dataclass
class Polyline(Figure):
    """Polyline"""

    def __init__(self, cnv: Canvas, points: Points, **kwargs) -> None:
        self.cnv = cnv
        self.points = points

        super().__init__(self.cnv, self.points, **kwargs)


@dataclass
class Line(Figure):
    """Line"""

    point = [0, 0]

    def __init__(self, cnv: Canvas, point: Point = None, **kwargs) -> None:
        self.cnv = cnv
        self.line = self.points = [self.point if point is None else point]

        super().__init__(self.cnv, self.points, **kwargs)

    def draw(
        self,
        start_point: Point,
        end_point: Point,
        stroke_width: Optional[Union[int, bool]] = False,
        stroke_color: Optional[Union[str, bool]] = False,
    ) -> None:
        """Draw line"""

        self.line = [end_point]
        self.points = [start_point, end_point]
        self.stroke_width = stroke_width if stroke_width else self.stroke_width
        self.stroke_color = stroke_color if stroke_color else self.stroke_color

        super().draw(None, self.stroke_width, self.stroke_color)

        return self

    def add(
        self, x: int, y: int, stroke_width: int = 0, stroke_color: str = None
    ) -> None:
        """Add line"""

        self.stroke_width = stroke_width if stroke_width else self.stroke_width
        self.stroke_color = stroke_color if stroke_color else self.stroke_color

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
        f"Converting HEX #ff0000[ff] (#f00[f]) to RGB is equal to: {Utils.hex_to_rgba(cfg.colors[11])}"
    )
    print("-" * 10)

    width = 200
    height = width
    axis_offset = width * 0.75
    rotation_angle = 45

    cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

    oval = PolyOval(cnv, width, height, offset_x=axis_offset, offset_y=axis_offset)
    rect = Rectangle(cnv, width, height, offset_x=axis_offset, offset_y=axis_offset)
    polyline = Polyline(
        cnv,
        [(0, 0.3), (-20, width / 2), (20, width / 2)],
        offset_x=axis_offset,
        offset_y=axis_offset,
    )
    line = Line(cnv)

    for i in range(0, cfg.height, 50):
        line.draw([0, i], [cfg.width, i], stroke_color=cfg.colors[6]).draw(
            [i, 0], [i, cfg.height]
        )

    for i, fig in enumerate([rect, oval, polyline]):
        # 1
        fig.draw(fill_color=cfg.colors[i % len(cfg.colors)])
        # 2
        fig.translate(250, 0).draw(
            stroke_width=5,
            stroke_color=cfg.colors[(i + 6) % len(cfg.colors)],
            fill_color=cfg.colors[i % len(cfg.colors)],
        )
        # 3
        fig.translate(250, 0).scale(1, 0.5).draw(
            stroke_color=cfg.colors[i % len(cfg.colors)],
            fill_color=(
                cfg.colors[(i + 1) % len(cfg.colors)]
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
            stroke_color=(cfg.colors[(i + 6) % len(cfg.colors)]),
            fill_color=(cfg.colors[i % len(cfg.colors)]),
        )
        # 7
        fig.reset().translate(150, 650).draw()
        # 8
        fig.set_pivot(0.5, 1).translate(250, 0).scale(0.5, 1).rotate(
            rotation_angle * -3
        ).draw(stroke_width=2)
        # 9
        fig.move(650, 650).draw(fill_color=cfg.colors[i % len(cfg.colors)]).set_pivot(
            0.5, 0.5
        )
        # 10
        fig.translate(-500, 250).scale(1, 0.5).rotate(-45).draw(
            stroke_color=cfg.colors[(i + 6) % len(cfg.colors)],
        )
        fig.translate(250, 0).rotate(-15).skew_x(15).skew_y(15).draw(
            stroke_color=cfg.colors[(i + 6) % len(cfg.colors)],
        )

    line = (
        Line(
            cnv,
            [axis_offset, axis_offset],
            stroke_color=cfg.colors[5],
        )
        .add(-axis_offset / 2, 0)
        .add(0, axis_offset / 4, 2, cfg.colors[6])
        .add(axis_offset / 4, axis_offset / 4, 4, cfg.colors[7])
    )
    print("Line points", line.line)

    cv.imshow("Common Canvas", cnv)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
