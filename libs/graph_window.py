""" Graph window """

import time, os

try:
    import tkinter as tk  # type:ignore
except ImportError:
    import Tkinter as tk  # type:ignore

__version__ = "1.0.0"


OBJ_ALREADY_DRAWN = "Object currently drawn"
UNSUPPORTED_METHOD = "Object doesn't support operation"
BAD_OPTION = "Illegal option value"

_root = tk.Tk()
_root.withdraw()

_update_lasttime = time.time()


def update(rate=None):
    """Update Time"""

    global _update_lasttime  # pylint: disable=W0603

    if rate:
        now = time.time()
        pause_length = 1 / rate - (now - _update_lasttime)

        if pause_length > 0:
            time.sleep(pause_length)
            _update_lasttime = now + pause_length
        else:
            _update_lasttime = now

    _root.update()


class GraphWin(tk.Canvas):
    """A GraphWin is a toplevel window for displaying graphics."""

    def __init__(
        self, width: int, height: int, title="Graphics Window", autoflush=True
    ):
        master = tk.Toplevel(_root)
        tk.Canvas.__init__(
            self, master, width=width, height=height, highlightthickness=0, bd=0
        )
        self.master.title(title)
        self.pack()
        self.mainloop()
        # self.height = int(height)
        # self.width = int(width)
        # if autoflush:
        #     _root.update()
        # _root.mainloop()

    def __repr__(self):
        return "GraphWin('{}', {}, {})".format(1, 2, 3)

    def __str__(self):
        return repr(self)


# def test():
#     """Test function"""

#     GraphWin(800, 600)


# if __name__ == "__main__":
#     test()
