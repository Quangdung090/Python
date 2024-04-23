from customtkinter import *
from PIL import Image
import csv

from component.lichSuItem import LichSuItem
from component.dialog import LichSuDialog

def CenterWindowToDisplay(Screen: CTk, width: int, height: int, scale_factor: float = 1.0):
        screen_width = Screen.winfo_screenwidth()
        screen_height = Screen.winfo_screenheight()
        x = int(((screen_width/2) - (width/2)) * scale_factor)
        y = int(((screen_height/2) - (height/1.85)) * scale_factor)
        return f"{width}x{height}+{x}+{y}"

class LichSu(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(width=1080, height=720, fg_color="#ffffff", corner_radius=0)
        self.pack_propagate(0)

        self.contentFrame = CTkScrollableFrame(
            master=self,
            width=1000,
            height=700,
            fg_color="#ffffff",
            corner_radius=0,
            scrollbar_fg_color="#ffffff",
        )
        self.contentFrame.place(x=40, y=20)
        self.items = []

    def initData(self):
        for i in self.items:
            i.pack_forget()
        self.items.clear()
        with open('history/history.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='|', quotechar='"')
            for row in spamreader:
                filepath = "history/images/" + row[0]
                img = Image.open(filepath)
                name = row[0].replace('--', ':')
                newItem = LichSuItem(self.contentFrame, self, img, filepath,name, row[1])
                self.items.append(newItem)
        for i in self.items:
            i.pack(anchor="n", pady=10)

    def showDialog(self, image, filepath, resultText):
        LichSuDialog(
            self.master,
            image,
            filepath,
            resultText
        )