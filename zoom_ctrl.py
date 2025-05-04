# requires: pyautogui pynput, pyqinauto keyboard
import pyautogui
import platform
from pynput.keyboard import Controller as KeyController, Key
import pygetwindow as gw


class zoom_interface:
    def __init__(self):
        self.os = platform.system().lower()
        from pynput.mouse import Controller as MouseController

        self.mouse = MouseController()
        self.kb = KeyController()

    def zoom(self, loc, mag):
        # Set center of zoom
        # self.zoom_center(loc)

        # Set zoom in, out, or no change
        if mag < 0:
            self.zoom_in()
        elif mag > 0:
            self.zoom_out()
        else:
            print("No Zoom")

    def zoom_center(
        self, loc
    ):  # This current method does not enable use of mouse, but only nerds use a mouse
        window = gw.getActiveWindow()
        if window:
            # Move the mouse to the desired location
            print("Title:", window.title)
            print("Position:", window.left, window.top)
            print("Size:", window.width, window.height)
            x, y = loc
            x = x * window.width + window.left
            y = y * window.height + window.top
            pyautogui.moveTo(x, y)
        else:
            print("No Active Window Found")

    def zoom_in(self):
        with self.kb.pressed(Key.ctrl):
            self.mouse.scroll(0, 2)  # Scroll up to zoom in
            print("Zoomed In")

    def zoom_out(self):
        with self.kb.pressed(Key.ctrl):
            self.mouse.scroll(0, -2)  # Scroll down to zoom out
            print("Zoomed Out")
