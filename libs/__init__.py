""" Libs module """

# pylint: disable=C0413

import sys
import pathlib

LIBS_PATH = pathlib.Path(__file__).parent.resolve()

sys.path.append(str(LIBS_PATH))

import figure_factory
import figure_factory_3d

__all__ = [*figure_factory.__all__, *figure_factory_3d.__all__]
