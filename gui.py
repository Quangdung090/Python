from customtkinter import *
from CTkTable import CTkTable
from PIL import Image
from trangchu import TrangChu
from lichsu import LichSu
import tkinter as tk

def CenterWindowToDisplay(Screen: CTk, width: int, height: int, scale_factor: float = 1.0):
        screen_width = Screen.winfo_screenwidth()
        screen_height = Screen.winfo_screenheight()
        x = int(((screen_width/2) - (width/2)) * scale_factor)
        y = int(((screen_height/2) - (height/1.85)) * scale_factor)
        return f"{width}x{height}+{x}+{y}"

class App(CTk):

    # global trangChu
    # global lichSu
    global trangChuBtn
    global lichSuBtn

    def __init__(self):
        super().__init__()
        set_appearance_mode("light")
        self.geometry(CenterWindowToDisplay(self, 1280, 720, self._get_window_scaling()))
        self.resizable(width=False, height=False)
        self.title("App")

        sideMenu = CTkFrame(
            master=self, 
            width=200, 
            height=720, 
            fg_color="#6a8eae", 
            corner_radius=0
        )
        sideMenu.pack_propagate(0)
        sideMenu.pack(fill="y", anchor="w", side="left")

        trangChuIcon_data = Image.open("icon/home.png")
        trangChuIcon = CTkImage(
            light_image=trangChuIcon_data, 
            dark_image=trangChuIcon_data
        )
        global trangChuBtn
        trangChuBtn = CTkButton(
            master=sideMenu, 
            width=200, 
            height=50, 
            corner_radius=1, 
            image=trangChuIcon,
            fg_color="#6a8eae", 
            text="Trang chủ",
            font=("Arial Bold", 16), 
            text_color="#FFFFFF", 
            command=lambda: self.show_frame(self.frames[TrangChu], trangChuBtn)
        )
        trangChuBtn.pack(anchor="center", pady=(70,0))


        lichSuIcon_data = Image.open("icon/history.png")
        lichSuIcon = CTkImage(
            light_image=lichSuIcon_data, 
            dark_image=lichSuIcon_data
        )
        global lichSuBtn
        lichSuBtn = CTkButton(
            master=sideMenu, 
            width=200, 
            height=50, 
            corner_radius=1, 
            image=lichSuIcon,
            fg_color="#6a8eae", 
            text="Lịch sử", 
            font=("Arial Bold",16), 
            text_color="#FFFFFF", 
            command=lambda: self.show_frame(self.frames[LichSu], lichSuBtn)
        )
        lichSuBtn.pack(anchor="center")

        self.frames = {}
        for F in (TrangChu, LichSu):
            frame = F(self)
            self.frames[F] = frame

        self.show_frame(self.frames[TrangChu], trangChuBtn)

    def show_frame(self, frame, btn):
        for F in self.frames:
            self.frames[F].pack_forget()
        for B in (trangChuBtn, lichSuBtn):
            B.configure(fg_color="#6a8eae")
        btn.configure(fg_color="#435b70")
        frame.pack(anchor="center")
        if(frame == self.frames[LichSu]):
            self.frames[LichSu].initData()


app = App()
app.mainloop()