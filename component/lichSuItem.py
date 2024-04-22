from customtkinter import *
from CTkTable import CTkTable
from PIL import Image

def CenterWindowToDisplay(Screen: CTk, width: int, height: int, scale_factor: float = 1.0):
        screen_width = Screen.winfo_screenwidth()
        screen_height = Screen.winfo_screenheight()
        x = int(((screen_width/2) - (width/2)) * scale_factor)
        y = int(((screen_height/2) - (height/1.85)) * scale_factor)
        return f"{width}x{height}+{x}+{y}"

class LichSuItem(CTkFrame):
    def __init__(self, master, lichSu, image, filepath, name, resultText, **kwargs):
        super().__init__(master, **kwargs)
        self.resultText = resultText
        self.lichSu = lichSu
        self.filepath=filepath
        self.configure(
            width=1000,
            height=150, 
            fg_color="#ffffff",
            border_color="#000000",
            border_width=1,
            corner_radius=0,
        )

        self.image = CTkImage(
            light_image=image,
            size=(148, 148)
        )
        self.imageLabel = CTkLabel(
            master=self,
            image=self.image,
            width=148,
            height=148,
            text=""
        )
        self.imageLabel.place(x=1, y=1)

        self.nameLabel = CTkLabel(
            master=self,
            text=name,
            width=848,
            height=148,
            font=("Arial Semi Bold", 16)
        )
        self.nameLabel.place(x=150, y=1)

        self.component_binding(self.imageLabel)
        self.component_binding(self.nameLabel)
        self.component_binding(self)


    def component_binding(self, component):
        component.bind("<Enter>", self.on_enter)
        component.bind("<Leave>", self.on_leave)
        component.bind("<Button>", self.showDialog)

    def on_enter(self, e):
        self.configure(
            fg_color="#dddddd"
        )

    def on_leave(self, e):
        self.configure(
            fg_color="#ffffff"
        )

    def showDialog(self, e):
        self.lichSu.showDialog(self.image, self.filepath, self.resultText)