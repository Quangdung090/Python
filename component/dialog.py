from customtkinter import * 
from PIL import Image
from mltu.transformers import ImageResizer
import cv2

def CenterWindowToDisplay(Screen: CTk, width: int, height: int, scale_factor: float = 1.0):
        screen_width = Screen.winfo_screenwidth()
        screen_height = Screen.winfo_screenheight()
        x = int(((screen_width/2) - (width/2)) * scale_factor)
        y = int(((screen_height/2) - (height/1.85)) * scale_factor)
        return f"{width}x{height}+{x}+{y}"

class LichSuDialog(CTkToplevel):
    def __init__(self, parent, image, filepath, resultText):
        super().__init__(parent)
        self.image = image
        self.filepath = filepath
        self.resultText = resultText
        self.geometry(CenterWindowToDisplay(self, 1000, 580, self._get_window_scaling()))
        self.master.attributes('-disabled', True)

        global imageLabel
        imageLabel = CTkLabel(
            master=self,
            font=('Arial Bold', 14),
            anchor="center",
            width=540,
            height=580,
            text="",
            fg_color="red"
        )
        imageLabel.place(x=0, y=0)


        inputImage = cv2.imread(filepath)
        displayImg = ImageResizer.resize_maintaining_aspect_ratio(
            inputImage,
            imageLabel.cget("width"),
            imageLabel.cget("height")
        )
        im = Image.fromarray(displayImg)
        imageCTk = CTkImage(
            light_image=im,
            size=(imageLabel.cget("width"), imageLabel.cget("height"))
        )
        imageLabel.configure(image=imageCTk)
        
        resultTextBox = CTkTextbox(
            master=self,
            width=460,
            height=580,
            font=("Arial Bold", 14),
            border_width=1,
            text_color="#000000",
        )
        resultTextBox.insert("0.0", self.resultText)
        resultTextBox.configure(state="disabled")
        resultTextBox.place(x=540, y=0)
    
    def destroy(self):
        self.master.attributes('-disabled', False)
        return super().destroy()