# inferenceModel.py
import cv2
import typing
import numpy as np
import language_tool_python
import tkinter as tk
import tkinter.font as tkFont
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as mb
from PIL import ImageTk, Image
from fpdf import FPDF
from pyaspeller import YandexSpeller
from tkinter.ttk import *
# from App import App


# from autocorrect import Speller
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer


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

        global GLabel_277
        GLabel_277=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_277["font"] = ft
        GLabel_277["fg"] = "#333333"
        GLabel_277["justify"] = "center"
        GLabel_277["text"] = "Image"
        GLabel_277.place(x=0,y=140,width=541,height=570)

        global GMessage_172
        Message=tk.Message(root)
        ft = tkFont.Font(family='Times',size=10)
        Message["font"] = ft
        Message["fg"] = "#333333"
        Message["justify"] = "center"
        Message["text"] = "Title"
        # GMessage_172.place(x=540,y=140,width=540,height=570)
        GMessage_172 = tk.Text(root, background=Message.cget("background"), relief="flat",
        borderwidth=0, font=Message.cget("font"), state="disabled",)
        GMessage_172.configure(state="normal")
        GMessage_172.insert(END,"title")
        GMessage_172.configure(state="disabled")
        GMessage_172.place(x=540,y=140,width=540,height=570)
        Message.destroy()


        global GButton_788
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
        GButton_788.place_forget()

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

        global GButton_656
        GButton_656=tk.Button(root)
        GButton_656["activebackground"] = "#618B25"
        GButton_656["activeforeground"] = "#fcfcfe"
        GButton_656["bg"] = "#6bd425"
        GButton_656["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_656["font"] = ft
        GButton_656["fg"] = "#ffffff"
        GButton_656["justify"] = "center"
        GButton_656["text"] = "Từng bước cắt ảnh \n(bước 1 làm xám ảnh)"
        GButton_656.place(x=20,y=20,width=150,height=45)
        GButton_656["command"] = self.GButton_656_command
        GButton_656.place_forget()
        
        global GButton_63
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
        GButton_63.place_forget()

        global GButton_29
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
        GButton_29.place_forget()

        global GButton_894
        GButton_894=tk.Button(root)
        GButton_894["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=10)
        GButton_894["font"] = ft
        GButton_894["fg"] = "#000000"
        GButton_894["justify"] = "center"
        GButton_894["text"] = "B3 Đóng khung"
        GButton_894.place(x=20,y=20,width=150,height=45)
        GButton_894["command"] = self.GButton_894_command
        GButton_894.place_forget()

        global GButton_249
        GButton_249=tk.Button(root)
        GButton_249["bg"] = "#f0f0f0"
        GButton_249["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_249["font"] = ft
        GButton_249["fg"] = "#000000"
        GButton_249["justify"] = "center"
        GButton_249["text"] = "Dự đoán từng câu"
        GButton_249.place(x=20,y=20,width=150,height=45)
        GButton_249["command"] = self.GButton_249_command
        GButton_249.place_forget()

        global GButton_385
        GButton_385=tk.Button(root)
        GButton_385["bg"] = "#f0f0f0"
        GButton_385["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_385["font"] = ft
        GButton_385["fg"] = "#000000"
        GButton_385["justify"] = "center"
        GButton_385["text"] = "Dự Đoán Toàn Bộ"
        GButton_385.place(x=200,y=20,width=150,height=45)
        GButton_385["command"] = self.GButton_385_command
        GButton_385.place_forget()

        global GButton_241
        GButton_241=tk.Button(root)
        GButton_241["bg"] = "#f0f0f0"
        GButton_241["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_241["font"] = ft
        GButton_241["fg"] = "#000000"
        GButton_241["justify"] = "center"
        GButton_241["text"] = "Hiển Thị lỗi"
        GButton_241.place(x=20,y=20,width=150,height=45)
        GButton_241["command"] = self.GButton_241_command
        GButton_241.place_forget()

        global GButton_120
        GButton_120=tk.Button(root)
        GButton_120["bg"] = "#f0f0f0"
        GButton_120["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_120["font"] = ft
        GButton_120["fg"] = "#000000"
        GButton_120["justify"] = "center"
        GButton_120["text"] = "Sửa Lỗi"
        GButton_120.place(x=200,y=20,width=150,height=45)
        GButton_120["command"] = self.GButton_120_command
        GButton_120.place_forget()

        global GButton_128
        GButton_128=tk.Button(root)
        GButton_128["bg"] = "#f0f0f0"
        GButton_128["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=10)
        GButton_128["font"] = ft
        GButton_128["fg"] = "#000000"
        GButton_128["justify"] = "center"
        GButton_128["text"] = "Tự chỉnh sửa thêm"
        GButton_128.place(x=20,y=20,width=150,height=45)
        GButton_128["command"] = self.GButton_128_command
        GButton_128.place_forget()

        global GButton_695
        GButton_695=tk.Button(root)
        GButton_695["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=10)
        GButton_695["font"] = ft
        GButton_695["fg"] = "#000000"
        GButton_695["justify"] = "center"
        GButton_695["text"] = "Lưu"
        GButton_695.place(x=20,y=20,width=150,height=45)
        GButton_695["command"] = self.GButton_695_command
        GButton_695.place_forget()

        global GButton_391
        GButton_391=tk.Button(root)
        GButton_391["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=10)
        GButton_391["font"] = ft
        GButton_391["fg"] = "#000000"
        GButton_391["justify"] = "center"
        GButton_391["text"] = "Hủy"
        GButton_391.place(x=200,y=20,width=150,height=45)
        GButton_391["command"] = self.GButton_391_command
        GButton_391.place_forget()

    def ChangeText(self,changeText):
        GMessage_172.configure(state="normal")
        GMessage_172.delete(1.0,END)
        GMessage_172.insert(END,changeText)
        GMessage_172.configure(state="disabled")

    def showImg(self,inputImage):
        displayImg = ImageResizer.resize_maintaining_aspect_ratio(inputImage,541,570)
        im = Image.fromarray(displayImg)
        imtk = ImageTk.PhotoImage(im)
        GLabel_277.configure(image=imtk)
        GLabel_277.image = imtk

    def GButton_845_command(self):
        print("Choose file to upload")
        # self.ChangeText("Loading...")
        UploadAction()
        GButton_788.place(x=20,y=20,width=150,height=45)
        
     

    def GButton_997_command(self):
        print("clear Text")
        self.ChangeText("")
        # Ẩn hết nút
        GButton_788.place_forget()
        GButton_63.place_forget()
        GButton_656.place_forget()
        GButton_29.place_forget()
        GButton_894.place_forget()
        GButton_249.place_forget()
        GButton_385.place_forget()
        GButton_120.place_forget()
        GButton_241.place_forget()
        GButton_128.place_forget()
        GButton_695.place_forget()
        GButton_391.place_forget()


    def GButton_470_command(self):
        print("clear Image")
        GLabel_277.configure(image="")
        # Ẩn hết nút
        GButton_788.place_forget()
        GButton_63.place_forget()
        GButton_656.place_forget()
        GButton_29.place_forget()
        GButton_894.place_forget()
        GButton_249.place_forget()
        GButton_385.place_forget()
        GButton_120.place_forget()
        GButton_241.place_forget()
        GButton_128.place_forget()
        GButton_695.place_forget()
        GButton_391.place_forget()


    def GButton_413_command(self):
        print("Save as pdf")
        saveAsPdf()


    def GButton_236_command(self):
        print("quit")
        showConfirmDialog()


    def GButton_788_command(self):
        print("Cắt ảnh")
        cropImageAction()
        GButton_656.place(x=20,y=20,width=150,height=45)
        GButton_63.place(x=200,y=20,width=150,height=45)

    def GButton_656_command(self):
        print("Cắt ảnh b1")
        imgGrayAction()
        GButton_29.place(x=20,y=20,width=150,height=45)


    def GButton_63_command(self):
        print("dự đoán")
        self.final_predict = ""
        GButton_249.place(x=20,y=20,width=150,height=45)
        GButton_385.place(x=200,y=20,width=150,height=45)
        app.showImg(boxImg)
        # predictAction()
        # GButton_63.place_forget()

    def GButton_29_command(self):
        print("Tô chữ")
        colorTextAction()
        GButton_894.place(x=20,y=20,width=150,height=45)

    def GButton_894_command(self):
        print("Đóng khung")
        boxesImgAction()
        GButton_894.place_forget()
        GButton_29.place_forget()

    def GButton_249_command(self):
        print("Dự đoán từng câu")
        PredictLineAction()


    def GButton_385_command(self):
        print("Dự đoán toàn bộ")
        PredictAllAction()
        GButton_241.place(x=20,y=20,width=150,height=45)
        GButton_120.place(x=200,y=20,width=150,height=45)

    def GButton_241_command(self):
        print("Hiển thị lỗi")
        showErrorAction()


    def GButton_120_command(self):
        print("Sửa Lỗi")
        self.pdfTextArray = []
        fixErrorAction()
        GButton_128.place(x=20,y=20,width=150,height=45)
        GButton_63.place_forget()
        GButton_120.place_forget()
        GButton_385.place_forget()

    def GButton_128_command(self):
        print("NGười dùng tự sửa thêm")
        GButton_695.place(x=20,y=20,width=150,height=45)
        GButton_391.place(x=200,y=20,width=150,height=45)
        GMessage_172.configure(state="normal")


    def GButton_695_command(self):
        print("Lưu")
        newOutput = GMessage_172.get(1.0,END)
        outputArr = newOutput.split('\n')
        self.pdfTextArray = outputArr
        GButton_695.place_forget()
        GButton_391.place_forget()
        GMessage_172.configure(state="disabled")


    def GButton_391_command(self):
        print("Hủy")
        GButton_695.place_forget()
        GButton_391.place_forget()
        GMessage_172.configure(state="disabled")
        
        
        

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        # print(*self.input_shapes[2])
        image = ImageResizer.resize_maintaining_aspect_ratio(image, 1408,96)
        # print(image)

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        # cv2.imshow('',image_pred)

        preds = self.model.run(None, {'input': image_pred})[0]

        # print(preds)

        text = ctc_decoder(preds, self.char_list)[0]
        # print(self.char_list)
        # text = "test"

        return text
    
def imgGrayAction():
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global thresh2
    ret, thresh2 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    app.showImg(thresh2)

def colorTextAction():
    global mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,2))
    mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)
    app.showImg(mask)

def waithere(milisecond=300):
    var = IntVar()
    root.after(milisecond, var.set, 1)
    print("waiting...")
    root.wait_variable(var)

def boxesImgAction():
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0,0,255), 2)
        waithere()
        app.showImg(bboxes_img)
        # app.showImg(bboxes_img)
        # cv2.imshow("",bboxes_img)
        # cv2.waitKey(0)
        bboxes.append((x,y,w,h))
    app.showImg(bboxes_img)
    

def cropSentence():
    imgGrayAction()
    colorTextAction()
    global boxImg
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    imgList = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0,0,255), 2)
        crop_img = img[y:y+h, x:x+w]
        # prediction_text = model.predict(crop_img)
        # cv2.putText(bboxes_img, prediction_text, (x, y -4), 0, 0.02*h, (30,15,252), ((w+h)//900)//3)
        # prediction.append(prediction_text)
        imgList.append(crop_img)
        # print("Prediction: ", prediction_text)
        # cv2.imshow(predictiont_tex,crop_img)
        # cv2.waitKey(0)
        bboxes.append((x,y,w,h))
    # cv2.imshow("",bboxes_img)
    # im = Image.fromarray(bboxes_img)
    # imtk = ImageTk.PhotoImage(im)
    # GLabel_277.configure(image=imtk)
    # GLabel_277.image = imtk
    app.showImg(bboxes_img)
    boxImg = bboxes_img
    return imgList

# def PredictionFormCropImg(Imglist):
#     prediction = []
#     for i in Imglist:
#         prediction_text = model.predict(i)
#         prediction.append(prediction_text)
#     return prediction

# def crop_text(img_path):
#     img = cv2.imread(img_path)
#   # Read in the image and convert to grayscale
#     img = img[:-20, :-20]  # Perform pre-cropping
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = 255*(gray < 50).astype(np.uint8)  # To invert the text to white
#     gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones(
#     (2, 2), dtype=np.uint8))  # Perform noise filtering
#     coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
#     x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
#     # Crop the image - note we do this on the original image
#     rect = img[y-2:y+h-4, x:x+w]
#     return rect


def error_correcting(text):
    tool = language_tool_python.LanguageTool('en-US')
    datasets = tool.correct(text)
    return datasets

def error_correct_pyspeller(sample_text):
    speller = YandexSpeller()
    fixed = speller.spelled(sample_text)
    return fixed

def wordSpliter(str,n=12):
    pieces = str.split()
    return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))

def PredictLineAction():
    if(len(imgList) <= 0):
        app.GButton_385_command()
        return
    currImg = imgList.pop()
    app.showImg(currImg)
    app.ChangeText("waitting...")
    waithere(100)
    predictText = model.predict(currImg)
    app.ChangeText(predictText)
    app.final_predict = app.final_predict + " " + predictText

def PredictAllAction():
    app.ChangeText("waitting...")
    waithere(100)
    while len(imgList) >0:
        currImg = imgList.pop()
        predictText = model.predict(currImg)
        app.final_predict = app.final_predict + " " + predictText
    ShowResult = wordSpliter(app.final_predict)
    app.showImg(boxImg)
    app.ChangeText("\n".join(ShowResult))

def showErrorAction():
    app.ChangeText("waitting...")
    waithere(100)
    print("start checking")
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(app.final_predict)

    resultStr =""
    for i in matches:
        resultStr += '\n'+ i.message + '\n' + i.context
        # resultStr += '\n{}\n{}\n'.format(
        # i.context, ' ' * (i.offsetInContext)+ '^' * i.errorLength
        # )
    app.ChangeText(resultStr)


def fixErrorAction():
    app.ChangeText("waitting...")
    waithere(100)
    print("start fixing")
    # output_text = error_correct_pyspeller(app.final_predict)
        # print(output_text)

    output_data = error_correcting(app.final_predict)
    result = wordSpliter(output_data)
    app.pdfTextArray = wordSpliter(output_data)
    app.ChangeText("\n".join(result))

def UploadAction(event=None):
    image_path = filedialog.askopenfilename()

    global img
    img = cv2.imread(image_path)
    app.showImg(img)

def cropImageAction():
    global imgList
    imgList = cropSentence()
    

def saveAsPdf(event=None):
        
    pdf = FPDF()
  
         #Add a page
    pdf.add_page()

         #set style and size of font 
         #that you want in the pdf
    pdf.set_font("Arial", size = 15)
        
        # pdf.cell(200,35,txt=output_data,ln=10,align='J')
    for x in app.pdfTextArray:
        pdf.cell(200, 10, txt = x, ln = 10, align = 'J') 
            
    pdf.output("output.pdf")     
    mb.showinfo("success","Save Pdf Successfully")

def showConfirmDialog(event=None):
    result = mb.askquestion('Exit Application', 'Do you really want to exit?') # mb = messagebox (import from tkinter)
        
    if(result == 'yes'):
        root.destroy()
    else:
        return

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/202301131202/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
 
    root = tk.Tk()
    app = App(root)
    root.mainloop()