"""Figure Factory 3D"""

# pylint: disable=E1101, W0237, R0913, R0902, C0301, R0913, R1705

from dataclasses import dataclass
from typing import Union, Tuple, List, Optional
import numpy as np
import cv2 as cv
from utils import Utils, RGB
from config import Config
from figure_factory import Rectangle, Polyline

__version__ = "1.0.0"

__all__ = []

Matrix = Tuple[
    Tuple[float, float, float, int],
    Tuple[float, float, float, int],
    Tuple[float, float, float, int],
    Tuple[int, int, int, int],
]
Point = Tuple[float, float]
Points = np.ndarray[Point]
Canvas = List[Tuple[int, int, int, Union[int, None]]]


@dataclass
class Figure3D:
    """Figure3D"""

    x: float = 0
    y: float = 0
    z: float = 0
    width: float = 0
    height: float = 0
    length: float = 0
    pivot: Point = None
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
        length: float = length,
        # **kwargs
        stroke_width: float = stroke_width,
        stroke_color: str = stroke_color,
        fill_color: str = fill_color,
    ) -> None:
        self.cnv = cnv
        self.width = width
        self.height = height
        self.length = length
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.fill_color = fill_color
        self.initial_props = self.points = points
        self.parts = []

    @staticmethod
    def get_projection_matrix_3d(length: float) -> Matrix:
        """Returns translate matrix"""

        return [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, length, 0),
            (0, 0, 0, 1),
        ]

    @staticmethod
    def get_translate_matrix_3d(tx: float, ty: float, tz: float) -> Matrix:
        """Returns translate matrix"""

        return [
            (1, 0, 0, tx),
            (0, 1, 0, ty),
            (0, 0, 1, tz),
            (1, 0, 0, 1),
        ]

    @staticmethod
    def get_scale_matrix_3d(sw: float, sh: float, sl: float) -> Matrix:
        """Returns scale matrix"""

        return [
            (sw, 0, 0, 0),
            (0, sh, 0, 0),
            (0, 0, sl, 0),
            (0, 0, 0, 1),
        ]

    @staticmethod
    def get_rotate_matrix_3d(rx: float, ry: float, rz: float) -> Matrix:
        """Returns rotate matrix"""

        rx = np.array(
            [
                (1, 0, 0, 0),
                (0, np.cos(rx), -np.sin(rx), 0),
                (0, np.sin(rx), np.cos(rx), 0),
                (0, 0, 0, 1),
            ]
        )

        ry = np.array(
            [
                (np.cos(ry), 0, np.sin(ry), 0),
                (0, 1, 0, 0),
                (-np.sin(ry), 0, np.cos(ry), 0),
                (0, 0, 0, 1),
            ]
        )

        rz = np.array(
            [
                (np.cos(rz), -np.sin(rz), 0, 0),
                (np.sin(rz), np.cos(rz), 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ]
        )

        return np.dot(np.dot(rz, ry), rx)

    @staticmethod
    def get_skew_matrix_3d(sx: float, sy: float, sz: float) -> Matrix:
        """Returns rotate matrix"""

        sx = np.array(
            [
                (1, np.tan(sx), 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ]
        )

        sy = np.array(
            [
                (1, 0, 0, 0),
                (np.tan(sy), 1, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 1),
            ]
        )

        sz = np.array(
            [
                (1, 0, np.tan(sz), 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ]
        )

        return np.dot(np.dot(sx, sy), sz)

    def _apply_projection_matrix_3d(self, points, matrix: Matrix) -> None:
        matrix = np.array(matrix).T
        points = np.hstack((points, np.ones((points.shape[0], 2))))
        points = np.dot(points, matrix)
        points = np.delete(points, -1, axis=1)

        return points

    def _apply_matrix_3d(self, matrix: Matrix) -> None:
        matrix = np.array(matrix).T
        points = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        points = np.dot(points, matrix)
        points = np.delete(points, -1, axis=1)

        self.points = points

    def _make_parallelepiped_parts(self) -> None:
        """Make parallelepiped parts"""

        back = self.points[: int(len(self.points) / 2)]
        front = self.points[int(len(self.points) / 2) :]
        bottom = [*back[2:4], *np.flip(front[2:4], axis=0)]
        top = [*back[0:2], *np.flip(front[:2], axis=0)]
        right = [*back[1:3], *np.flip(front[1:3], axis=0)]
        left = [back[0], front[0], front[3], back[3]]

        self.parts = np.array([back, bottom, left, right, front, top])

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

    def translate_3d(self, tx: float, ty: float, tz: float) -> "Figure3D":
        """Translate"""

        self.x += tx
        self.y += ty
        self.z += tz

        self._apply_matrix_3d(Figure3D.get_translate_matrix_3d(tx, ty, tz))

        self._make_parallelepiped_parts()

        return self

    def scale_3d(self, sw: float, sh: float, sl: float) -> "Figure3D":
        """Scale"""

        self._apply_matrix_3d(
            Figure3D.get_translate_matrix_3d(-self.x, -self.y, -self.z)
        )
        self._apply_matrix_3d(
            Figure3D.get_scale_matrix_3d(
                Utils.deg_to_rads(sw),
                Utils.deg_to_rads(sh),
                Utils.deg_to_rads(sl),
            )
        )
        self._apply_matrix_3d(Figure3D.get_translate_matrix_3d(self.x, self.y, self.z))

        self._make_parallelepiped_parts()

        return self

    def rotate_3d(self, rx: float, ry: float, rz: float) -> "Figure3D":
        """Rotate"""

        self._apply_matrix_3d(
            Figure3D.get_translate_matrix_3d(-self.x, -self.y, -self.z)
        )
        self._apply_matrix_3d(
            Figure3D.get_rotate_matrix_3d(
                Utils.deg_to_rads(rx),
                Utils.deg_to_rads(ry),
                Utils.deg_to_rads(rz),
            )
        )
        self._apply_matrix_3d(Figure3D.get_translate_matrix_3d(self.x, self.y, self.z))

        self._make_parallelepiped_parts()

        return self

    def skew_3d(self, sx: float, sy: float, sz: float) -> "Figure3D":
        """Skew"""

        self._apply_matrix_3d(
            Figure3D.get_translate_matrix_3d(-self.x, -self.y, -self.z)
        )
        self._apply_matrix_3d(
            Figure3D.get_skew_matrix_3d(
                Utils.deg_to_rads(sx),
                Utils.deg_to_rads(sy),
                Utils.deg_to_rads(sz),
            )
        )
        self._apply_matrix_3d(Figure3D.get_translate_matrix_3d(self.x, self.y, self.z))

        self._make_parallelepiped_parts()

        return self

    def draw(
        self,
        stroke_width: Union[float, bool] = False,
        stroke_color: Union[str, bool] = False,
        fill_color: Union[str, bool] = False,
    ) -> "Figure3D":
        """Draw figure"""

        stroke_width = self._get_stroke_width(stroke_width)
        stroke_color = self._get_stroke_color(stroke_color)
        fill_color = self._get_fill_color(fill_color)

        for points in self.parts:
            points = np.delete(points, -1, axis=1)

            Polyline(
                self.cnv,
                points,
                stroke_width=self.stroke_width,
                stroke_color=self.stroke_color,
            ).draw()

        return self


@dataclass
class Parallelepiped(Figure3D):
    """Parallelepiped"""

    def __init__(  # pylint: disable=R0914
        self, cnv: Canvas, width, height, length, **kwargs
    ) -> None:
        Rectangle(cnv, width, height)

        back = Rectangle(cnv, width, height)
        front = Rectangle(cnv, width, height)

        bsp = self._apply_projection_matrix_3d(
            back.points, Figure3D.get_projection_matrix_3d(-length * 0.5)
        )

        fsp = self._apply_projection_matrix_3d(
            front.points, Figure3D.get_projection_matrix_3d(length * 0.5)
        )

        points = np.concatenate((bsp, fsp), axis=0)

        super().__init__(cnv, points, width, height, length, **kwargs)

        self._make_parallelepiped_parts()


def test():
    """Test function"""

    cfg = Config()

    cnv = np.full(cfg.cnv_props, 255, dtype=np.uint8)

    prllppd_config = 400, 350, 250  # width, height, length
    cx = cfg.width / 2
    cy = cfg.height / 2
    win_name = "Common Canvas"

    cfg.grid(cnv, color=Utils.hex_to_rgba(cfg.colors[15]), position=(int(cx), int(cy)))
    prllppd = (
        Parallelepiped(cnv, *prllppd_config, stroke_width=2, stroke_color=cfg.colors[0])
        .translate_3d(cx, cy, prllppd_config[2] / 2)
        .rotate_3d(45, 45, 45)
    )
    prllppd.draw()

    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, cnv)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    test()
