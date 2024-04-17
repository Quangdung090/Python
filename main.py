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
from tkinter import ttk
from PIL import ImageTk, Image
from fpdf import FPDF
from pyaspeller import YandexSpeller
from tkinter.ttk import *
# from App import App


# from autocorrect import Speller
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

class HoverButton(tk.Button):
    def __init__(self, master, bg="", hoverbg=""):
        tk.Button.__init__(self,master=master,)
        self.bg = bg
        self.hoverbg = hoverbg
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = self.hoverbg

    def on_leave(self, e):
        self['background'] = self.bg

class App:        
    
    def __init__(self, root):
        #setting title
        root.title("undefined")
        #setting window size
        width=1080
        height=710
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.title("Hand-writing recoginzer")
        root.resizable(width=False, height=False)

        upLoadBtn=HoverButton(root,"#6bd425","#618B25")
        upLoadBtn["activebackground"] = "#6bd425"
        upLoadBtn["activeforeground"] = "#fcfcfc"
        upLoadBtn["bg"] = "#6bd425"
        upLoadBtn["cursor"] = "hand2"
        upLoadBtn["disabledforeground"] = "#ba8888"
        ft = tkFont.Font(family='Times',size=12, weight="bold")
        upLoadBtn["relief"]="ridge"
        upLoadBtn["font"] = ft
        upLoadBtn["fg"] = "#eeeeee"
        upLoadBtn["justify"] = "center"
        upLoadBtn["text"] = "Choose file to upload"
        upLoadBtn.place(x=20,y=20,width=150,height=45)
        upLoadBtn["command"] = self.upLoadBtn_command

        clearTextBtn=HoverButton(root,"#f0f7ee","#8c8b8b")
        clearTextBtn["activebackground"] = "#8c8b8b"
        clearTextBtn["activeforeground"] = "#ffffff"
        clearTextBtn["bg"] = "#f0f7ee"
        clearTextBtn["relief"]="ridge"
        clearTextBtn["cursor"] = "hand2"
        clearTextBtn["disabledforeground"] = "#9f8181"
        ft = tkFont.Font(family='Times',size=12, weight="bold")
        clearTextBtn["font"] = ft
        clearTextBtn["fg"] = "#000000"
        clearTextBtn["justify"] = "center"
        clearTextBtn["text"] = "Clear text"
        clearTextBtn.place(x=900,y=90,width=150,height=45)
        clearTextBtn["command"] = self.clearTextBtn_command

        savePDFBtn=HoverButton(root, "#5bc0be", "#1F7A8C")
        savePDFBtn["activebackground"] = "#5bc0be"
        savePDFBtn["activeforeground"] = "#ffffff"
        savePDFBtn["bg"] = "#5bc0be"
        savePDFBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=12, weight="bold")
        savePDFBtn["relief"]="ridge"
        savePDFBtn["font"] = ft
        savePDFBtn["fg"] = "#fffefe"
        savePDFBtn["justify"] = "center"
        savePDFBtn["text"] = "Save As Pdf"
        savePDFBtn.place(x=200,y=20,width=150,height=45)
        savePDFBtn["command"] = self.savePDFBtn_command

        exitAppBtn=HoverButton(root, "#e9190f", "#FB3640")
        exitAppBtn["activebackground"] = "#e9190f"
        exitAppBtn["activeforeground"] = "#ffffff"
        exitAppBtn["bg"] = "#e9190f"
        exitAppBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        exitAppBtn["font"] = ft
        exitAppBtn["fg"] = "#fefefe"
        exitAppBtn["justify"] = "center"
        exitAppBtn["text"] = "EXIT"
        exitAppBtn["relief"] = "ridge"
        exitAppBtn.place(x=900,y=20,width=150,height=45)
        exitAppBtn["command"] = self.exitAppBtn_command

        global imageLabel
        imageLabel=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        imageLabel["font"] = ft
        imageLabel["fg"] = "#333333"
        imageLabel["justify"] = "center"
        imageLabel["text"] = "Click on 'Choose file to upload' to put image here"
        imageLabel.place(x=0,y=140,width=541,height=570)

        global resultText
        Message=tk.Message(root)
        ft = tkFont.Font(family='Comic Sans MS', weight="bold",size=14)
        Message["font"] = ft
        Message["fg"] = "#333333"
        Message["justify"] = "center"
        Message["text"] = "Your text goes here"
        # resultText.place(x=540,y=140,width=540,height=570)
        resultText = tk.Text(root, background="white", relief="solid",
        borderwidth=1,font=Message.cget("font"), state="disabled")
        resultText.configure(state="normal")
        resultText.insert(END,"Your text goes here")
        resultText.configure(state="disabled")
        resultText.place(x=540,y=140,width=540,height=570)
        Message.destroy()


        global catAnhBtn
        catAnhBtn=HoverButton(root,"#6bd425","#618B25")
        catAnhBtn["activebackground"] = "#6bd425"
        catAnhBtn["bg"] = "#6bd425"
        ft = tkFont.Font(family='Times',size=12, weight="bold")
        catAnhBtn['cursor'] = "hand2"
        catAnhBtn["font"] = ft
        catAnhBtn["relief"]="ridge"
        catAnhBtn["fg"] = "#ffffff"
        catAnhBtn["justify"] = "center"
        catAnhBtn["text"] = "Cắt ảnh"
        catAnhBtn.place(x=20,y=20,width=150,height=45)
        catAnhBtn["command"] = self.catAnhBtn_command
        catAnhBtn.place_forget()
        
        clearImgBtn=HoverButton(root,"#f0f7ee","#8c8b8b")
        clearImgBtn["activebackground"] = "#8c8b8b"
        clearImgBtn["activeforeground"] = "#ffffff"
        clearImgBtn["bg"] = "#f0f7ee"
        clearImgBtn["relief"]="ridge"
        clearImgBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=12, weight="bold")
        clearImgBtn["font"] = ft
        clearImgBtn["fg"] = "#000000"
        clearImgBtn["justify"] = "center"
        clearImgBtn["text"] = "Clear Image"
        clearImgBtn.place(x=20,y=90,width=150,height=45)
        clearImgBtn["command"] = self.clearImgBtn_command

        global lamXamAnhBtn
        lamXamAnhBtn=HoverButton(root,"#6bd425","#618B25")
        lamXamAnhBtn["activebackground"] = "#618B25"
        lamXamAnhBtn["activeforeground"] = "#fcfcfe"
        lamXamAnhBtn["bg"] = "#6bd425"
        lamXamAnhBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=10, weight="bold")
        lamXamAnhBtn["font"] = ft
        lamXamAnhBtn["fg"] = "#ffffff"
        lamXamAnhBtn["relief"]="ridge"
        lamXamAnhBtn["justify"] = "center"
        lamXamAnhBtn["text"] = "Từng bước cắt ảnh \n(bước 1 làm xám ảnh)"
        lamXamAnhBtn.place(x=20,y=20,width=150,height=45)
        lamXamAnhBtn["command"] = self.lamXamAnhBtn_command
        lamXamAnhBtn.place_forget()
        
        global duDoanChuBtn
        duDoanChuBtn=HoverButton(root,"#5bc0be","#1F7A8C")
        duDoanChuBtn["activebackground"] = "#5bc0be"
        duDoanChuBtn["activeforeground"] = "#ffffff"
        duDoanChuBtn["bg"] = "#5bc0be"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        duDoanChuBtn["font"] = ft
        duDoanChuBtn["relief"]="ridge"
        duDoanChuBtn["fg"] = "#ffffff"
        duDoanChuBtn["justify"] = "center"
        duDoanChuBtn["text"] = "Dự đoán chữ"
        duDoanChuBtn.place(x=200,y=20,width=150,height=45)
        duDoanChuBtn["command"] = self.duDoanChuBtn_command
        duDoanChuBtn.place_forget()

        global toChuBtn
        toChuBtn=HoverButton(root,"#6bd425","#618B25")
        toChuBtn["activebackground"] = "#6bd425"
        toChuBtn["activeforeground"] = "#fcfcfc"
        toChuBtn["bg"] = "#6bd425"
        toChuBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        toChuBtn["relief"]="ridge"
        toChuBtn["font"] = ft
        toChuBtn["fg"] = "#ffffff"
        toChuBtn["justify"] = "center"
        toChuBtn["text"] = "B2 Tô Chữ"
        toChuBtn.place(x=20,y=20,width=150,height=45)
        toChuBtn["command"] = self.toChuBtn_command
        toChuBtn.place_forget()

        global dongKhungBtn
        dongKhungBtn=HoverButton(root,"#6bd425","#618B25")
        dongKhungBtn["activebackground"] = "#6bd425"
        dongKhungBtn["activeforeground"] = "#fcfcfc"
        dongKhungBtn["bg"] = "#6bd425"
        dongKhungBtn["fg"] = "#ffffff"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        dongKhungBtn["relief"]="ridge"
        dongKhungBtn["font"] = ft
        dongKhungBtn["justify"] = "center"
        dongKhungBtn["text"] = "B3 Đóng khung"
        dongKhungBtn.place(x=20,y=20,width=150,height=45)
        dongKhungBtn["command"] = self.dongKhungBtn_command
        dongKhungBtn.place_forget()

        global duDoanTungcauBtn
        duDoanTungcauBtn=HoverButton(root,"#6bd425","#618B25")
        duDoanTungcauBtn["activebackground"] = "#6bd425"
        duDoanTungcauBtn["activeforeground"] = "#fcfcfc"
        duDoanTungcauBtn["bg"] = "#6bd425"
        duDoanTungcauBtn["fg"] = "#ffffff"
        duDoanTungcauBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        duDoanTungcauBtn["relief"]="ridge"
        duDoanTungcauBtn["font"] = ft
        duDoanTungcauBtn["justify"] = "center"
        duDoanTungcauBtn["text"] = "Dự đoán từng câu"
        duDoanTungcauBtn.place(x=20,y=20,width=150,height=45)
        duDoanTungcauBtn["command"] = self.duDoanTungcauBtn_command
        duDoanTungcauBtn.place_forget()

        global duDoanToanBoBtn
        duDoanToanBoBtn=HoverButton(root,"#5bc0be","#1F7A8C")
        duDoanToanBoBtn["activebackground"] = "#5bc0be"
        duDoanToanBoBtn["activeforeground"] = "#ffffff"
        duDoanToanBoBtn["bg"] = "#5bc0be"
        duDoanToanBoBtn["fg"] = "#fffefe"
        duDoanToanBoBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        duDoanToanBoBtn["relief"]="ridge"
        duDoanToanBoBtn["font"] = ft
        duDoanToanBoBtn["justify"] = "center"
        duDoanToanBoBtn["text"] = "Dự Đoán Toàn Bộ"
        duDoanToanBoBtn.place(x=200,y=20,width=150,height=45)
        duDoanToanBoBtn["command"] = self.duDoanToanBoBtn_command
        duDoanToanBoBtn.place_forget()

        global showErorrBtn
        showErorrBtn=HoverButton(root,"#6bd425","#618B25")
        showErorrBtn["activebackground"] = "#6bd425"
        showErorrBtn["activeforeground"] = "#fcfcfc"
        showErorrBtn["bg"] = "#6bd425"
        showErorrBtn["fg"] = "#ffffff"
        showErorrBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        showErorrBtn["relief"]="ridge"
        showErorrBtn["font"] = ft
        showErorrBtn["justify"] = "center"
        showErorrBtn["text"] = "Hiển Thị lỗi"
        showErorrBtn.place(x=20,y=20,width=150,height=45)
        showErorrBtn["command"] = self.showErorrBtn_command
        showErorrBtn.place_forget()

        global fixErrorBtn
        fixErrorBtn=HoverButton(root,"#5bc0be","#1F7A8C")
        fixErrorBtn["activebackground"] = "#5bc0be"
        fixErrorBtn["activeforeground"] = "#ffffff"
        fixErrorBtn["bg"] = "#5bc0be"
        fixErrorBtn["fg"] = "#fffefe"
        fixErrorBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        fixErrorBtn["relief"]="ridge"
        fixErrorBtn["font"] = ft
        fixErrorBtn["justify"] = "center"
        fixErrorBtn["text"] = "Sửa Lỗi"
        fixErrorBtn.place(x=200,y=20,width=150,height=45)
        fixErrorBtn["command"] = self.fixErrorBtn_command
        fixErrorBtn.place_forget()

        global editTextBtn
        editTextBtn=HoverButton(root,"#6bd425","#618B25")
        editTextBtn["activebackground"] = "#6bd425"
        editTextBtn["activeforeground"] = "#fcfcfc"
        editTextBtn["bg"] = "#6bd425"
        editTextBtn["fg"] = "#ffffff"
        editTextBtn["cursor"] = "hand2"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        editTextBtn["relief"]="ridge"
        editTextBtn["font"] = ft
        editTextBtn["justify"] = "center"
        editTextBtn["text"] = "Tự chỉnh sửa thêm"
        editTextBtn.place(x=20,y=20,width=150,height=45)
        editTextBtn["command"] = self.editTextBtn_command
        editTextBtn.place_forget()

        global saveEditedTextBtn
        saveEditedTextBtn=HoverButton(root,"#6bd425","#618B25")
        saveEditedTextBtn["activebackground"] = "#6bd425"
        saveEditedTextBtn["activeforeground"] = "#fcfcfc"
        saveEditedTextBtn["bg"] = "#6bd425"
        saveEditedTextBtn["fg"] = "#ffffff"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        saveEditedTextBtn["relief"]="ridge"
        saveEditedTextBtn["font"] = ft
        saveEditedTextBtn["justify"] = "center"
        saveEditedTextBtn["text"] = "Lưu"
        saveEditedTextBtn.place(x=20,y=20,width=150,height=45)
        saveEditedTextBtn["command"] = self.saveEditedTextBtn_command
        saveEditedTextBtn.place_forget()

        global cancelEditedTextBtn
        cancelEditedTextBtn=HoverButton(root,"#e9190f","#FB3640")
        cancelEditedTextBtn["activebackground"] = "#e9190f"
        cancelEditedTextBtn["activeforeground"] = "#ffffff"
        cancelEditedTextBtn["bg"] = "#e9190f"
        cancelEditedTextBtn["fg"] = "#ffffff"
        ft = tkFont.Font(family='Times',size=14, weight="bold")
        cancelEditedTextBtn["relief"]="ridge"
        cancelEditedTextBtn["font"] = ft
        cancelEditedTextBtn["justify"] = "center"
        cancelEditedTextBtn["text"] = "Hủy"
        cancelEditedTextBtn.place(x=200,y=20,width=150,height=45)
        cancelEditedTextBtn["command"] = self.cancelEditedTextBtn_command
        cancelEditedTextBtn.place_forget()
    

    def ChangeText(self,changeText):
        resultText.configure(state="normal")
        resultText.delete(1.0,END)
        resultText.insert(END,changeText)
        resultText.configure(state="disabled")

    def showImg(self,inputImage):
        displayImg = ImageResizer.resize_maintaining_aspect_ratio(inputImage,imageLabel.winfo_width(),imageLabel.winfo_height())
        im = Image.fromarray(displayImg)
        imtk = ImageTk.PhotoImage(im)
        imageLabel.configure(image=imtk)
        imageLabel.image = imtk

    def resetDisplay(self):
        catAnhBtn.place_forget()
        duDoanChuBtn.place_forget()
        lamXamAnhBtn.place_forget()
        toChuBtn.place_forget()
        dongKhungBtn.place_forget()
        duDoanTungcauBtn.place_forget()
        duDoanToanBoBtn.place_forget()
        fixErrorBtn.place_forget()
        showErorrBtn.place_forget()
        editTextBtn.place_forget()
        saveEditedTextBtn.place_forget()
        cancelEditedTextBtn.place_forget()
        resultText.configure(state="disabled")



    def upLoadBtn_command(self):
        print("Choose file to upload")
        # self.ChangeText("Loading...")
        UploadAction()
        catAnhBtn.place(x=20,y=20,width=150,height=45)     

    def clearTextBtn_command(self):
        print("clear Text")
        output = resultText.get(1.0,END)
        if(output != "Your text goes here\n"):
            self.ChangeText("")
            # Ẩn hết nút
            self.resetDisplay()

    def clearImgBtn_command(self):
        print("clear Image")
        imageLabel.configure(image="")
        # Ẩn hết nút
        self.resetDisplay()


    def savePDFBtn_command(self):
        print("Save as pdf")
        saveAsPdf()


    def exitAppBtn_command(self):
        print("quit")
        showConfirmDialog()


    def catAnhBtn_command(self):
        print("Cắt ảnh")
        cropImageAction()
        lamXamAnhBtn.place(x=20,y=20,width=150,height=45)
        duDoanChuBtn.place(x=200,y=20,width=150,height=45)

    def lamXamAnhBtn_command(self):
        print("Cắt ảnh b1")
        imgGrayAction()
        toChuBtn.place(x=20,y=20,width=150,height=45)


    def duDoanChuBtn_command(self):
        print("dự đoán")
        self.final_predict = ""
        duDoanTungcauBtn.place(x=20,y=20,width=150,height=45)
        duDoanToanBoBtn.place(x=200,y=20,width=150,height=45)
        app.showImg(boxImg)
        # predictAction()
        # duDoanChuBtn.place_forget()

    def toChuBtn_command(self):
        print("Tô chữ")
        colorTextAction()
        dongKhungBtn.place(x=20,y=20,width=150,height=45)

    def dongKhungBtn_command(self):
        print("Đóng khung")
        boxesImgAction()
        dongKhungBtn.place_forget()
        toChuBtn.place_forget()

    def duDoanTungcauBtn_command(self):
        print("Dự đoán từng câu")
        PredictLineAction()


    def duDoanToanBoBtn_command(self):
        print("Dự đoán toàn bộ")
        PredictAllAction()
        showErorrBtn.place(x=20,y=20,width=150,height=45)
        fixErrorBtn.place(x=200,y=20,width=150,height=45)

    def showErorrBtn_command(self):
        print("Hiển thị lỗi")
        showErrorAction()


    def fixErrorBtn_command(self):
        print("Sửa Lỗi")
        self.pdfTextArray = []
        self.result_txt =""
        fixErrorAction()
        editTextBtn.place(x=20,y=20,width=150,height=45)
        duDoanChuBtn.place_forget()
        fixErrorBtn.place_forget()
        duDoanToanBoBtn.place_forget()

    def editTextBtn_command(self):
        print("NGười dùng tự sửa thêm")
        saveEditedTextBtn.place(x=20,y=20,width=150,height=45)
        cancelEditedTextBtn.place(x=200,y=20,width=150,height=45)
        resultText.configure(state="normal")


    def saveEditedTextBtn_command(self):
        print("Lưu")
        newOutput = resultText.get(1.0,END)
        outputArr = newOutput.split('\n')
        self.pdfTextArray = outputArr
        self.result_txt = newOutput
        saveEditedTextBtn.place_forget()
        cancelEditedTextBtn.place_forget()
        resultText.configure(state="disabled")


    def cancelEditedTextBtn_command(self):
        print("Hủy")
        self.ChangeText(self.result_txt)
        saveEditedTextBtn.place_forget()
        cancelEditedTextBtn.place_forget()
        resultText.configure(state="disabled")
        
        
        

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
    # imageLabel.configure(image=imtk)
    # imageLabel.image = imtk
    app.showImg(bboxes_img)
    boxImg = bboxes_img
    return imgList

def error_correcting(text):
    tool = language_tool_python.LanguageTool('en-US')
    datasets = tool.correct(text)
    return datasets

def wordSpliter(str,n=10):
    pieces = str.split()
    return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))

def PredictLineAction():
    if(len(imgList) <= 0):
        app.duDoanToanBoBtn_command()
        return
    currImg = imgList.pop()
    app.showImg(currImg)
    app.ChangeText("waiting...")
    waithere(100)
    predictText = model.predict(currImg)
    app.ChangeText(predictText)
    app.final_predict = app.final_predict + " " + predictText

def PredictAllAction():
    app.ChangeText("waiting...")
    waithere(100)
    while len(imgList) >0:
        currImg = imgList.pop()
        predictText = model.predict(currImg)
        app.final_predict = app.final_predict + " " + predictText
    ShowResult = wordSpliter(app.final_predict)
    app.showImg(boxImg)
    app.ChangeText("\n".join(ShowResult))

def showErrorAction():
    app.ChangeText("waiting...")
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
    app.ChangeText("waiting...")
    waithere(100)
    print("start fixing")
    # output_text = error_correct_pyspeller(app.final_predict)
        # print(output_text)

    output_data = error_correcting(app.final_predict)
    result = wordSpliter(output_data)
    app.pdfTextArray = wordSpliter(output_data)
    app.result_txt = "\n".join(result)
    app.ChangeText(app.result_txt)

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