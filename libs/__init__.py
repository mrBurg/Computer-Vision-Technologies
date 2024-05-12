""" Libs module """

import sys
import pathlib

LIBS_PATH = pathlib.Path(__file__).parent.resolve()
print(LIBS_PATH)

sys.path.append(str(LIBS_PATH))

# from figure_factory import *
# import figure_factory as ff

# __all__ = [*ff.__all__]
# __all__ = ff
