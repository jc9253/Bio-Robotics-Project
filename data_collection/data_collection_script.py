import cv2
import tkinter as tk
from tkinter import messagebox
import random
import time
import threading
import platform

# Platform-specific setup
if platform.system() == "Windows":
    from ctypes import windll
    windll.user32.SetProcessDPIAware()

class Text():
    def __init__(self, subject):
        self.word = "Hello"
        self.subject = subject
    
    def _start_root(self):
    def __init__(self, subject):
        self.word = "Hello"
        self.subject = subject
    
    def _start_root(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main window
        self.window = tk.Toplevel(self.root)
        self.window.overrideredirect(True)

    def prep(self):
        self._start_root()
        
        self.size = random.randint(1, 100)
        self.x = random.randint(0, self.root.winfo_screenwidth() - self.size * len(self.word))
        self.y = random.randint(0, self.root.winfo_screenheight() - self.size)

        self.size_control = 100
        self.x_control = int((self.root.winfo_screenwidth() - self.size_control * len(self.word)) / 2)
        self.y_control = int((self.root.winfo_screenheight() - self.size_control) / 2)

        self.size_control = 100
        self.x_control = int((self.root.winfo_screenwidth() - self.size_control * len(self.word)) / 2)
        self.y_control = int((self.root.winfo_screenheight() - self.size_control) / 2)

        return f'{self.size}_{self.x}_{self.y}'
    
    def show_control(self):
        self.window.geometry(f"{self.size_control * len(self.word)}x{self.size_control}+{self.x_control}+{self.y_control}")

        label = tk.Label(self.window, text=self.word, font=("Arial", self.size_control))
        label.pack()

        # self.root.after(2000, self.window.destroy)  # Destroy after 2 seconds

        self.root.after(2000, self.root.destroy)
        self.root.mainloop()
    
    def show_control(self):
        self.window.geometry(f"{self.size_control * len(self.word)}x{self.size_control}+{self.x_control}+{self.y_control}")

        label = tk.Label(self.window, text=self.word, font=("Arial", self.size_control))
        label.pack()

        # self.root.after(2000, self.window.destroy)  # Destroy after 2 seconds

        self.root.after(2000, self.root.destroy)
        self.root.mainloop()

    def show(self):
        self._start_root()
        self.window.geometry(f"{self.size * len(self.word)}x{self.size}+{self.x}+{self.y}")
        self._start_root()
        self.window.geometry(f"{self.size * len(self.word)}x{self.size}+{self.x}+{self.y}")

        label = tk.Label(self.window, text=self.word, font=("Arial", self.size))
        label = tk.Label(self.window, text=self.word, font=("Arial", self.size))
        label.pack()

        self.root.after(2000, self.window.destroy)  # Destroy after 2 seconds
        self.root.after(2000, self.window.destroy)  # Destroy after 2 seconds

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_text = f"Timestamp: {timestamp}, Location: ({self.x}, {self.y}), Size: {self.size}\n"
        # print(log_text)

        with open(self.subject + "log.txt", "a") as log:
        with open(self.subject + "log.txt", "a") as log:
            log.write(log_text)

        self.root.after(2000, self.root.destroy)
        self.root.mainloop()

class Video:
    def __init__(self, path):
        try: 
            self.stop()
        except:
            pass
        self.path = path
        self.cap = None
        self.out = None
        self.recording = False
        self.thread = None

    def _record(self):
        while self.recording:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)
                # cv2.imshow("Recording", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def start(self, file_name):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access the webcam")
            return
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(self.path + file_name + ".avi", fourcc, 20.0, (640, 480))
        self.recording = True
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()

    def stop(self):
        self.recording = False
        if self.thread:
            self.thread.join()

        if self.out:
            self.out.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

class Photo():
    def __init__(self):
        pass
    
    def take(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access the webcam")
            return
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        ret, frame = cap.read()
        if ret:
            filename = f"photo_{timestamp.replace(':', '-')}.png"
            cv2.imwrite(filename, frame)
            # print(f"Photo saved as {filename}")

            with open("log.txt", "a") as log:
                log.write(f"Photo taken at {timestamp}\n")

            cv2.imshow("Photo", frame)
            cv2.waitKey(2000)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    subject = "Braley_"
    media = Video(path="videos/" + subject)
    display = Text(subject)
    display = Text(subject)

    print("Starting session")
    try: 
        while True:
            label = display.prep()
            media.start(label)
            
            display.show_control()
            # time.sleep(3)

            display.show()
            time.sleep(2)

            media.stop()


    except KeyboardInterrupt:
        print("Complete")
