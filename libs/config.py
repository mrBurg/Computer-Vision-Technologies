"""Config"""

import dataclasses
from typing import List, Optional, Union, Tuple
from utils import Utils

__version__ = "1.0.0"

CanvasProps = Tuple[int, Union[Tuple[int, Union[int, None]], None]]


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
    colors = default_colors

    def __init__(
        self,
        *props: CanvasProps,
        depth: Optional[int] = None,
        colors: Optional[List[str]] = None,  # Replaces default colors
        add_colors: Optional[List[str]] = None,  # Adds a set to the default colors
    ) -> None:
        self.width = self.default_cnv_props[1]
        self.height = self.default_cnv_props[0]
        self.depth = max(1, min(depth if depth else self.default_cnv_props[2], 4))

        if colors:
            self.colors = colors

        if add_colors:
            self.colors.extend(add_colors)

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
                    "colors",
                ]
            ],
            [
                f"{prop}: {str(getattr(self, prop, None))}"
                for prop in ["width", "height", "depth"]
            ],
        )

    def __str__(self) -> str:
        return repr(self)
