"""Utils"""

# pylint: disable=C0301

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Callable, Tuple, Union, List, Optional
from math import radians, degrees

__version__ = "2.0.0"

RGB = Tuple[int, int, int, Union[float, None]]
BGR = RGB


@dataclass
class Utils:
    """Utils"""

    @staticmethod
    def reshuffle(rgb: RGB) -> BGR:
        """Rearrangement"""

        if len(rgb) == 3:
            return rgb[::-1]

        rgb_channels = rgb[:3]

        return (*rgb_channels[::-1], rgb[-1])

    @staticmethod
    def hex_to_rgba(hex_str: str) -> RGB:
        """Converts a HEX color to RGB"""

        if hex_str:
            rgba = []
            hex_str = hex_str.lstrip("#")

            if len(hex_str) in (3, 4):
                return Utils.hex_to_rgba(f"#{''.join([hex * 2 for hex in hex_str])}")

            if len(hex_str) == 6:
                for i in range(0, 5, 2):
                    rgba.append(int(hex_str[i : i + 2], 16))

            elif len(hex_str) == 8:
                for i in range(0, 7, 2):
                    color = int(hex_str[i : i + 2], 16)

                    if i == 6:
                        color = 1 / 255 * color
                        rgba.append(color)

                        break

                    rgba.append(color)

            return tuple(rgba)

        return None

    @staticmethod
    def rgba_to_hex(r: int, g: int, b: int) -> str:
        """Converts a RGB color to HEX"""

        return f"#{r:02x}{g:02x}{b:02x}"  # "#%02x%02x%02x" % (r, g, b)

    @staticmethod
    def deg_to_rads(deg: float) -> float:
        """Converts degrees to radians"""

        return radians(deg)  # deg * np.pi / 180

    @staticmethod
    def rads_to_deg(rad: float) -> float:
        """Converts radians to degrees"""

        return degrees(rad)  # rad * 180 / np.pi

    @staticmethod
    def animate(animation: Callable[[], None], speed=0.01) -> float:
        """Makes animation"""

        while animation():
            time.sleep(speed)

    @staticmethod
    def description(
        name: str, props: Optional[List[str]] = None, args: Optional[List[str]] = None
    ) -> float:
        """Makes descripton for object"""

        description = f"\n{name} ->"

        if props and len(props):
            description += "\n props:\n  " + "\n  ".join(props)

        if args and len(args):
            description += "\n args:\n  " + "\n  ".join(args)

        return description + "\n"

    @staticmethod
    def get_path(path: str, flag: Optional[str] = None) -> str:
        """Get path"""

        # Needs TODO: relative r, absolute a, parent p

        files = Path.cwd().glob(f"**/{path}")
        files = list(files)

        if flag == "All":
            return files

        if files:
            return str(list(files)[0])

        return None


def test() -> None:
    """Test function"""

    color_rgba = "#f00f"
    color_rgb = "#f00"
    deg = 45

    print("\033[33m")
    print(
        f"HEX #ff0000[ff] or #f00[f] to RGB is equal to: {Utils.hex_to_rgba(color_rgba)} or {Utils.hex_to_rgba(color_rgb)}"
    )
    print(
        f"RGB {Utils.hex_to_rgba(color_rgb)} to HEX is equal to: {Utils.rgba_to_hex(*Utils.hex_to_rgba(color_rgb))}"
    )
    print(
        f"RGBA {Utils.hex_to_rgba(color_rgba)} to BGRA is equal to: {Utils.reshuffle(Utils.hex_to_rgba(color_rgba))}"
    )

    print(f"{deg} degrees in radians is equal to: {Utils.deg_to_rads(deg)}")
    print(
        f"{Utils.deg_to_rads(deg)} radians in degrees is equal to: {Utils.rads_to_deg(Utils.deg_to_rads(deg))}"
    )
    print("\033[0m")


if __name__ == "__main__":
    test()
