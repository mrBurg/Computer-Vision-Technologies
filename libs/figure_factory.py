"""Figure Factory"""

# pylint: disable=E1101, W0237, R0913, R0902, R0911, R1705, C0301

from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional
import numpy as np
import cv2 as cv
from utils import Utils, RGB
from config import Config

__version__ = "3.1.0"

__all__ = []

Matrix = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[int, int, int],
]
Point = Tuple[float, float]
Points = np.ndarray[Point]
Canvas = List[Tuple[int, int, int, Union[int, None]]]


@dataclass
class Figure:
    """Figure"""

    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    pivot: Point = field(default_factory=lambda: [0.5, 0.5])
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
        # **kwargs
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
        self.pivot = [0.5, 0.5]

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
        points = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        points = np.dot(points, matrix)
        points = np.delete(points, -1, axis=1)  # points2[:, [0, 1]]

        self.points = points

    def _get_stroke_width(
        self, weight: Optional[Union[int, float, bool]]
    ) -> Union[int, None]:
        """Get weight"""

        if isinstance(weight, bool):
            if self.stroke_width:
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

    def _get_stroke_color(self, color: Optional[Union[str | bool]]) -> RGB | None:
        """Get color"""

        if isinstance(color, bool):
            if self.stroke_color:
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

    def _get_fill_color(self, color: Optional[Union[str | bool]]) -> RGB | None:
        """Get color"""

        if isinstance(color, bool):
            if self.fill_color:
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

    def set_pivot(self, px: float, py: Optional[float] = None) -> "Figure":
        """Set Pivot"""

        py = px if py is None else py
        self.pivot = [px, py]

        self._apply_matrix(
            Figure.get_translate_matrix(
                self.width * 0.5 - self.pivot[0] * self.width,
                self.height * 0.5 - self.pivot[1] * self.height,
            )
        )

        return self

    def translate(self, tx: float, ty: Optional[float] = None) -> "Figure":
        """Translate"""

        ty = tx if ty is None else ty
        self.x += tx
        self.y += ty
        self._apply_matrix(Figure.get_translate_matrix(tx, ty))

        return self

    def scale(self, sw: float, sh: Optional[float] = None) -> "Figure":
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

    def move(self, mx: float, my: Optional[float] = None) -> "Figure":
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
        matrix: Optional[Matrix] = None,
        stroke_width: Optional[Union[int, float, bool]] = False,
        stroke_color: Optional[Union[str | bool]] = False,
        fill_color: Optional[Union[str | bool]] = False,
    ) -> "Figure":
        """Draw figure"""

        if matrix:
            self._apply_matrix(self.get_translate_matrix(-self.x, -self.y))
            self._apply_matrix(matrix)
            self._apply_matrix(self.get_translate_matrix(self.x, self.y))
            self.x += matrix[0][2]
            self.y += matrix[1][2]

        stroke_width = self._get_stroke_width(stroke_width)
        stroke_color = self._get_stroke_color(stroke_color)
        fill_color = self._get_fill_color(fill_color)

        if isinstance(self, Line):
            points = np.array([self.points[0], self.points[1]], dtype=np.int16)

            cv.line(
                self.cnv,
                points[0],
                points[1],
                stroke_color,
                1 if stroke_width < 1 else stroke_width,
            )
        else:
            points = np.array([self.points], dtype=np.int32)

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

        self.x, self.y = self.initial_coords
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
        offset_x: float = 0,
        offset_y: float = 0,
        **kwargs,
    ) -> None:
        self.start = start
        self.end = end
        self.quality = quality
        self.points = np.array(
            [
                (
                    np.sin(Utils.deg_to_rads(degree)) * width * 0.5,
                    np.cos(Utils.deg_to_rads(degree)) * height * 0.5,
                )
                for degree in range(
                    self.start, self.end, round(20 * (1 - self.quality) + self.quality)
                )
            ]
        )
        self.initial_coords = (offset_x, offset_y)

        self.translate(*self.initial_coords)

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

    def __init__(
        self,
        cnv: Canvas,
        width: float,
        height: float,
        offset_x: float = 0,
        offset_y: float = 0,
        **kwargs,
    ) -> None:
        self.cnv = cnv

        p1 = (width / -2, height / -2)
        p2 = (p1[0] + width, p1[1])
        p3 = (p2[0], p1[1] + height)
        p4 = (p1[0], p3[1])

        self.points = np.array([p1, p2, p3, p4])
        self.initial_coords = (offset_x, offset_y)

        self.translate(*self.initial_coords)

        super().__init__(self.cnv, self.points, width, height, **kwargs)


@dataclass
class Polyline(Figure):
    """Polyline"""

    def __init__(
        self,
        cnv: Canvas,
        points: Points,
        offset_x: float = 0,
        offset_y: float = 0,
        **kwargs,
    ) -> None:
        self.cnv = cnv
        self.points = np.array(points)
        self.initial_coords = (offset_x, offset_y)

        self.translate(*self.initial_coords)

        super().__init__(self.cnv, self.points, **kwargs)


@dataclass
class Line(Figure):
    """Line"""

    point = [0, 0]

    def __init__(self, cnv: Canvas, point: Optional[Point] = None, **kwargs) -> None:
        self.cnv = cnv
        self.line = self.points = np.array([self.point if point is None else point])

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
        self.points = np.array([start_point, end_point])
        self.stroke_width = stroke_width if stroke_width else self.stroke_width
        self.stroke_color = stroke_color if stroke_color else self.stroke_color

        super().draw(None, self.stroke_width, self.stroke_color)

        return self

    def add(
        self, x: int, y: int, stroke_width: int = 0, stroke_color: Optional[str] = None
    ) -> None:
        """Add line"""

        self.stroke_width = stroke_width if stroke_width else self.stroke_width
        self.stroke_color = stroke_color if stroke_color else self.stroke_color

        p1 = self.line[-1]
        p2 = [p1[0] + x, p1[1] + y]

        self.line = np.append(self.line, [p2], axis=0)
        self.points = np.array([p1, p2])

        super().draw(None, self.stroke_width, self.stroke_color)

        return self


def test():
    """Test function"""

    width = 200
    height = width
    axis_offset = width * 0.75
    rotation_angle = 45

    cfg = Config(768, 1024)

    cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

    oval = PolyOval(cnv, width, height, offset_x=axis_offset, offset_y=axis_offset)
    rect = Rectangle(cnv, width, height, offset_x=axis_offset, offset_y=axis_offset)
    polyline = Polyline(
        cnv,
        [(0, 0.3), (-20, width * 0.5), (20, width * 0.5)],
        offset_x=axis_offset,
        offset_y=axis_offset,
    )
    pivot = Oval(cnv, 10, 10, fill_color=cfg.colors[0])

    cfg.grid(cnv, color=Utils.hex_to_rgba(cfg.colors[6]))

    for i, fig in enumerate([rect, oval, polyline]):
        # 1
        fig.draw(fill_color=cfg.colors[i % len(cfg.colors)])
        pivot.move(fig.x, fig.y).draw()
        # 2
        fig.set_pivot(0.40, 0.40).translate(250, 0).draw(
            stroke_width=5,
            stroke_color=cfg.colors[(i + 6) % len(cfg.colors)],
            fill_color=cfg.colors[i % len(cfg.colors)],
        )
        pivot.move(fig.x, fig.y).draw()
        # 3
        fig.set_pivot(0.60, 0.60).translate(250, 0).scale(1, 0.5).draw(
            stroke_color=cfg.colors[i % len(cfg.colors)],
            fill_color=(
                cfg.colors[(i + 1) % len(cfg.colors)]
                if isinstance(fig, Polyline)
                else None
            ),
        )
        pivot.move(fig.x, fig.y).draw()
        # 4
        fig.translate(-500, 250).rotate(-rotation_angle).draw(
            stroke_color=None, fill_color=None
        )
        pivot.move(fig.x, fig.y).draw()
        # 5
        fig.translate(250, 0).rotate(rotation_angle).scale(0.5, 2).draw(
            stroke_width=None
        )
        pivot.move(fig.x, fig.y).draw()
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
        pivot.move(fig.x, fig.y).draw()
        # 7
        fig.reset().translate(0, 500).draw()
        pivot.move(fig.x, fig.y).draw()
        # 8
        fig.translate(250, 0).scale(0.5, 1).rotate(rotation_angle * -3).draw(
            stroke_width=2
        )
        pivot.move(fig.x, fig.y).draw()
        # 9
        fig.move(650, 650).rotate(-15).draw(fill_color=cfg.colors[i % len(cfg.colors)])
        pivot.move(fig.x, fig.y).draw()
        # 10
        fig.translate(-500, 250).scale(1, 0.5).rotate(-45).draw(
            stroke_color=cfg.colors[(i + 6) % len(cfg.colors)],
        )
        pivot.move(fig.x, fig.y).draw()
        # 11
        fig.translate(250, 0).rotate(-45).skew_x(15).skew_y(15).draw(
            stroke_color=cfg.colors[(i + 6) % len(cfg.colors)],
        )
        pivot.move(fig.x, fig.y).draw()

    line = (
        Line(
            cnv,
            [axis_offset, axis_offset],
            stroke_color=cfg.colors[5],
        )
        .add(-axis_offset * 0.5, 0)
        .add(0, axis_offset / 4, 2, cfg.colors[6])
        .add(axis_offset / 4, axis_offset / 4, 4, cfg.colors[7])
    )
    print("\033[33m")
    print("Line points", line.line)
    print("\033[0m")

    cv.imshow("Common Canvas", cnv)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
