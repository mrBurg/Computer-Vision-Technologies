"""Figure Factory 3D"""

# pylint: disable=E1101, W0237, R0913, R0902, C0301

from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional
import numpy as np
import cv2 as cv
from utils import Utils
from config import Config
from figure_factory import Figure, Rectangle, Canvas, Point, Points

__version__ = "1.0.0"

__all__ = ["Utils", "Config"]

Matrix = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[int, int, int, int],
]


@dataclass
class Figure3D(Figure):
    """Figure3D"""

    x: float = 0
    y: float = 0
    z: float = 0
    width: float = 0
    height: float = 0
    length: float = 0
    pivot: Point = field(default_factory=lambda: [0.5, 0.5, 0.5])
    initial_props: Points = None
    parts: List[List[Points]] = field(default_factory=lambda: [[[]]])

    def __init__(
        self,
        cnv: Canvas,
        points: Points,
        width: float = width,
        height: float = height,
        length: float = length,
        **kwargs
    ) -> None:
        self.cnv = cnv
        self.width = width
        self.height = height
        self.length = length
        self.initial_props = self.points = points
        self.pivot = [0.5, 0.5, 0.5]
        offset_x = kwargs.get("offset_x", self.x)
        offset_y = kwargs.get("offset_y", self.y)
        self.z = kwargs.pop("offset_z", self.z)
        self.parts = [[[]]]

        self.translate(offset_x, offset_y)

        super().__init__(self.cnv, self.points, self.width, self.height, **kwargs)

    @staticmethod
    def get_projection_matrix_3d() -> Matrix:
        """Returns translate matrix"""

        return [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 1),
        ]

    @staticmethod
    def get_translate_matrix_3d(tx: float, ty: float, tz: float) -> Matrix:
        """Returns translate matrix"""

        return [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [1, 0, 0, 1],
        ]

    # @staticmethod
    # def get_scale_matrix(sw: float, sh: float) -> Matrix:
    #     """Returns scale matrix"""

    #     return [
    #         (sw, 0, 0),
    #         (0, sh, 0),
    #         (0, 0, 1),
    #     ]

    @staticmethod
    def get_rotate_x_matrix_3d(rad: float) -> Matrix:
        """Returns rotate matrix"""

        return [
            [1, 0, 0, 0],
            [0, np.cos(rad), np.sin(rad), 0],
            [0, -np.sin(rad), np.cos(rad), 0],
            [0, 0, 0, 1],
        ]

    # @staticmethod
    # def get_skew_x_matrix(deg: float) -> Matrix:
    #     """Returns skewX matrix"""

    #     return [
    #         (1, np.tan(deg), 0),
    #         (0, 1, 0),
    #         (0, 0, 1),
    #     ]

    # @staticmethod
    # def get_skew_y_matrix(deg: float) -> Matrix:
    #     """Returns skewY matrix"""

    #     return [
    #         (1, 0, 0),
    #         (np.tan(deg), 1, 0),
    #         (0, 0, 1),
    #     ]

    def _make_parts(self) -> None:
        print(self.points)

        back = self.points[: int(len(self.points) / 2)]
        front = self.points[int(len(self.points) / 2) :]
        bottom = np.array([*back[2:4], *np.flip(front[2:4], axis=0)])
        top = np.array([*back[0:2], *np.flip(front[:2], axis=0)])
        right = np.array([*back[1:3], *np.flip(front[1:3], axis=0)])
        left = np.array([back[0], front[0], front[3], back[3]])

        self.parts = np.array([back, bottom, right, left, top, front])

    def _apply_matrix_3d(self, matrix: Matrix) -> None:
        matrix = np.array(matrix).T
        points = np.array([]).reshape(0, 2)

        # print(self.points)
        # print(matrix)
        # print(points)

        for p in self.points:
            point = np.dot([*p, 0, 1], matrix)
            points = np.append(points, [point[:2]], axis=0)

        self.points = points

    # def set_pivot_3d(self, px: float, py: float = None, pz: float = None) -> "Figure3D":
    #     """Set Pivot"""

    #     py = px if py is None else py
    #     self.pivot = [px, py]

    #     self._apply_matrix(
    #         Figure3D.get_translate_matrix(
    #             self.width * 0.5 - self.pivot[0] * self.width,
    #             self.height * 0.5 - self.pivot[1] * self.height,
    #         )
    #     )

    #     return self

    def translate_3d(self, tx: float, ty: float, tz: float) -> "Figure3D":
        """Translate"""

        self.x += tx
        self.y += ty
        self.z += tz
        # self._apply_matrix_3d(Figure3D.get_translate_matrix_3d(tx, ty, tz))

        return self

    # def scale(self, sw: float, sh: float = None) -> "Figure3D":
    #     """Scale"""

    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(-self.x, -self.y))

    #     sh = sw if sh is None else sh
    #     self._apply_matrix(Figure3D.get_scale_matrix(sw, sh))

    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(self.x, self.y))

    #     return self

    def rotate_x_3d(self, deg: float) -> "Figure3D":
        """Rotate"""

        self._apply_matrix_3d(Figure3D.get_translate_matrix_3d(-self.x, -self.y, 0))
        self._apply_matrix_3d(Figure3D.get_rotate_x_matrix_3d(Utils.deg_to_rads(deg)))
        self._apply_matrix_3d(Figure3D.get_translate_matrix_3d(self.x, self.y, 0))

        return self

    # def skew_x(self, deg: float) -> "Figure3D":
    #     """skew X"""

    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(-self.x, -self.y))
    #     self._apply_matrix(Figure3D.get_skew_x_matrix_3d(Utils.deg_to_rads(deg)))
    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(self.x, self.y))

    #     return self

    # def skew_y(self, deg: float) -> "Figure3D":
    #     """skew Y"""

    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(-self.x, -self.y))
    #     self._apply_matrix(Figure3D.get_skew_y_matrix_3d(Utils.deg_to_rads(deg)))
    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(self.x, self.y))

    #     return self

    # def move(self, mx: float, my: float = None) -> "Figure3D":
    #     """Move"""

    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(-self.x, -self.y))

    #     my = mx if my is None else my
    #     self.x = mx
    #     self.y = my
    #     self._apply_matrix(Figure3D.get_translate_matrix_3d(self.x, self.y))

    #     return self

    def draw(
        self,
        matrix: Matrix = None,
        _stroke_width: Optional[Union[int, float, bool]] = False,
        _stroke_color: Optional[Union[str | bool]] = False,
        _fill_color: Optional[Union[str | bool]] = False,
    ) -> "Figure3D":
        """Draw figure"""

        if matrix:
            self._apply_matrix_3d(
                self.get_translate_matrix_3d(-self.x, -self.y, -self.z)
            )
            self._apply_matrix_3d(matrix)
            self._apply_matrix_3d(self.get_translate_matrix_3d(self.x, self.y, self.z))
            self.x += matrix[0][3]
            self.y += matrix[1][3]
            self.z += matrix[2][3]

        self._make_parts()

        print(self.parts)
        print("-" * 20)

        for points in self.parts:
            super().__init__(
                self.cnv,
                points,
                self.width,
                self.height,
                self.stroke_width,
                self.stroke_color,
                self.fill_color,
            )
            super().draw()

        return self

    def reset(self) -> "Figure3D":
        """Reset"""

        # Needs TODO: Reset the size when given new

        self.x = self.y = self.z = 0
        self.stroke_width = 1
        self.stroke_color = self.fill_color = None
        self.points = self.initial_props

        return self


@dataclass
class Parallelepiped(Figure3D):
    """Parallelepiped"""

    def __init__(  # pylint: disable=R0914
        self, cnv: Canvas, width, height, length, **kwargs
    ) -> None:
        self.cnv = cnv

        offset_x = kwargs.pop("offset_x", 0)
        offset_y = kwargs.pop("offset_y", 0)
        offset_z = kwargs.pop("offset_z", 0)

        back_side = Rectangle(
            cnv, width, height, offset_x=offset_x, offset_y=offset_y
        ).draw()
        front_side = (
            Rectangle(cnv, width, height, offset_x=offset_x, offset_y=offset_y)
            .translate(length, length)
            .draw()
        )

        self.points = np.array([back_side.points, front_side.points]).reshape(-1, 2)
        self.initial_coords = (offset_x, offset_y, offset_z)

        super().__init__(self.cnv, self.points, width, height, length, **kwargs)


def test():
    """Test function"""

    cfg = Config()

    cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

    prllppd_config = 200, 100, 50
    cx = cfg.width / 2
    cy = cfg.height / 2

    cfg.grid(cnv, color=Utils.hex_to_rgba(cfg.colors[15]))

    prllppd = Parallelepiped(
        cnv,
        *prllppd_config,
        offset_x=cx,
        offset_y=cy,
        offset_z=prllppd_config[2],
        stroke_width=2,
        stroke_color=cfg.colors[0],
        # fill_color=cfg.colors[1],
    )
    prllppd.draw()

    cv.imshow("Common Canvas", cnv)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
