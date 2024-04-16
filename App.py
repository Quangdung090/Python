import tkinter as tk
import tkinter.font as tkFont

class App:
    def __init__(self, root):
        #setting title
        root.title("undefined")
        #setting window size
        width=1080
        height=720
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GButton_845=tk.Button(root)
        GButton_845["activebackground"] = "#618B25"
        GButton_845["activeforeground"] = "#fcfcfc"
        GButton_845["bg"] = "#6bd425"
        GButton_845["cursor"] = "arrow"
        GButton_845["disabledforeground"] = "#ba8888"
        ft = tkFont.Font(family='Times',size=12)
        GButton_845["font"] = ft
        GButton_845["fg"] = "#ffffff"
        GButton_845["justify"] = "center"
        GButton_845["text"] = "Choose file to upload"
        GButton_845.place(x=20,y=20,width=150,height=45)
        GButton_845["command"] = self.GButton_845_command

        GButton_997=tk.Button(root)
        GButton_997["activebackground"] = "#8c8b8b"
        GButton_997["activeforeground"] = "#15ff4b"
        GButton_997["bg"] = "#f0f7ee"
        GButton_997["cursor"] = "arrow"
        GButton_997["disabledforeground"] = "#9f8181"
        ft = tkFont.Font(family='Times',size=12)
        GButton_997["font"] = ft
        GButton_997["fg"] = "#000000"
        GButton_997["justify"] = "center"
        GButton_997["text"] = "Clear text"
        GButton_997.place(x=900,y=90,width=150,height=45)
        GButton_997["command"] = self.GButton_997_command

        GButton_413=tk.Button(root)
        GButton_413["activebackground"] = "#1F7A8C"
        GButton_413["activeforeground"] = "#15ff4b"
        GButton_413["bg"] = "#5bc0be"
        GButton_413["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=12)
        GButton_413["font"] = ft
        GButton_413["fg"] = "#fffefe"
        GButton_413["justify"] = "center"
        GButton_413["text"] = "Save As Pdf"
        GButton_413.place(x=200,y=20,width=150,height=45)
        GButton_413["command"] = self.GButton_413_command

        GButton_236=tk.Button(root)
        GButton_236["activebackground"] = "#FB3640"
        GButton_236["activeforeground"] = "#ffffff"
        GButton_236["bg"] = "#e9190f"
        GButton_236["cursor"] = "mouse"
        ft = tkFont.Font(family='Times',size=14)
        GButton_236["font"] = ft
        GButton_236["fg"] = "#fefefe"
        GButton_236["justify"] = "center"
        GButton_236["text"] = "EXIT"
        GButton_236["relief"] = "ridge"
        GButton_236.place(x=900,y=20,width=150,height=45)
        GButton_236["command"] = self.GButton_236_command

        GLabel_277=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_277["font"] = ft
        GLabel_277["fg"] = "#333333"
        GLabel_277["justify"] = "center"
        GLabel_277["text"] = "Image"
        GLabel_277.place(x=0,y=140,width=541,height=570)

        GMessage_172=tk.Message(root)
        ft = tkFont.Font(family='Times',size=10)
        GMessage_172["font"] = ft
        GMessage_172["fg"] = "#333333"
        GMessage_172["justify"] = "center"
        GMessage_172["text"] = "Title"
        GMessage_172.place(x=540,y=140,width=540,height=570)

        GButton_788=tk.Button(root)
        GButton_788["activebackground"] = "#618B25"
        GButton_788["bg"] = "#6bd425"
        ft = tkFont.Font(family='Times',size=12)
        GButton_788["font"] = ft
        GButton_788["fg"] = "#ffffff"
        GButton_788["justify"] = "center"
        GButton_788["text"] = "Cắt ảnh"
        GButton_788.place(x=20,y=20,width=150,height=45)
        GButton_788["command"] = self.GButton_788_command

        GButton_470=tk.Button(root)
        GButton_470["activebackground"] = "#8c8b8b"
        GButton_470["activeforeground"] = "#15ff4b"
        GButton_470["bg"] = "#f0f7ee"
        GButton_470["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=12)
        GButton_470["font"] = ft
        GButton_470["fg"] = "#000000"
        GButton_470["justify"] = "center"
        GButton_470["text"] = "Clear Image"
        GButton_470.place(x=20,y=90,width=150,height=45)
        GButton_470["command"] = self.GButton_470_command

        GButton_656=tk.Button(root)
        GButton_656["activebackground"] = "#618B25"
        GButton_656["activeforeground"] = "#fcfcfe"
        GButton_656["bg"] = "#6bd425"
        GButton_656["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_656["font"] = ft
        GButton_656["fg"] = "#ffffff"
        GButton_656["justify"] = "center"
        GButton_656["text"] = "HIển thị từng bước cắt ảnh (bước 1 làm xám ảnh)"
        GButton_656.place(x=20,y=20,width=150,height=45)
        GButton_656["command"] = self.GButton_656_command

        GButton_63=tk.Button(root)
        GButton_63["activebackground"] = "#1F7A8C"
        GButton_63["activeforeground"] = "#15ff4b"
        GButton_63["bg"] = "#5bc0be"
        ft = tkFont.Font(family='Times',size=12)
        GButton_63["font"] = ft
        GButton_63["fg"] = "#ffffff"
        GButton_63["justify"] = "center"
        GButton_63["text"] = "Dự đoán chữ"
        GButton_63.place(x=200,y=20,width=150,height=45)
        GButton_63["command"] = self.GButton_63_command

        GButton_29=tk.Button(root)
        GButton_29["bg"] = "#f0f0f0"
        GButton_29["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_29["font"] = ft
        GButton_29["fg"] = "#000000"
        GButton_29["justify"] = "center"
        GButton_29["text"] = "B2 Tô Chữ"
        GButton_29.place(x=20,y=20,width=150,height=45)
        GButton_29["command"] = self.GButton_29_command

        GButton_894=tk.Button(root)
        GButton_894["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=10)
        GButton_894["font"] = ft
        GButton_894["fg"] = "#000000"
        GButton_894["justify"] = "center"
        GButton_894["text"] = "B3 Đóng khung"
        GButton_894.place(x=20,y=20,width=150,height=45)
        GButton_894["command"] = self.GButton_894_command

    def GButton_845_command(self):
        print("command")


    def GButton_997_command(self):
        print("command")


    def GButton_413_command(self):
        print("command")


    def GButton_236_command(self):
        print("command")


    def GButton_788_command(self):
        print("command")


    def GButton_470_command(self):
        print("command")


    def GButton_656_command(self):
        print("command")


    def GButton_63_command(self):
        print("command")


    def GButton_29_command(self):
        print("command")


    def GButton_894_command(self):
        print("command")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
