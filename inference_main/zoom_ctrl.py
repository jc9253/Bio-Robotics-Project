#requires: pyautogui pynput, pyqinauto keyboard
import pyautogui
import platform
from pynput.keyboard import Controller as KeyController, Key

class zoom_interface ():
    def __init__(self):
        self.os = platform.system().lower()
        from pynput.mouse import Controller as MouseController
        self.mouse = MouseController()
        self.kb = KeyController()

    def zoom(self, loc, mag):
        # Set center of zoom
        self.zoom_center(loc)

        # Set zoom in, out, or no change
        if (mag < 0):
            self.zoom_in()
        elif (mag >= 0):
            self.zoom_out()
            
    def zoom_center(self, loc):
        # Move the mouse to the desired location
        x, y = loc
        pyautogui.moveTo(x, y)

    def zoom_in(self):
        with self.kb.pressed(Key.ctrl):
            self.mouse.scroll(0, 2)  # Scroll up to zoom in

    def zoom_out(self):
         with self.kb.pressed(Key.ctrl):
            self.mouse.scroll(0, -2)  # Scroll down to zoom out