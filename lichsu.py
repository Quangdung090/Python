from customtkinter import *
from CTkTable import CTkTable
from PIL import Image

class LichSu(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(width=1080, height=720, fg_color="blue", corner_radius=0)
