

class zoom ():
    def __init(self):
        pass

    def zoom(self, loc, mag):
        # Set center of zoom
        self.zoom_center(loc)

        # Set zoom in, out, or no change
        if (mag < 0):
            self.zoom_in()
        elif (mag >= 0):
            self.zoom_out()


class win_os(zoom):
    def __init__(self):
        # import pywinauto
        return NotImplemented
    
    def zoom_center(self, loc):
        return NotImplemented
    
    def zoom_in(self):
        return NotImplemented
    
    def zoom_out(self):
        return NotImplemented
    
        # pywinauto.keyboard.send_keys("{VK_CONTROL down}")
        # pywinauto.keyboard.send_keys("{+ down}")
        # pywinauto.keyboard.send_keys("{+ up}")
        # pywinauto.keyboard.send_keys("{VK_CONTROL up}")

class mac_os(zoom):
    def __init__(self):
        return NotImplemented
    
    def zoom_center(self, loc):
        return NotImplemented
    
    def zoom_in(self):
        return NotImplemented
    
    def zoom_out(self):
        return NotImplemented

class lin_os(zoom):
    def __init__(self):
        return NotImplemented
    
    def zoom_center(self, loc):
        return NotImplemented
    
    def zoom_in(self):
        return NotImplemented

    def zoom_out(self):
        return NotImplemented
