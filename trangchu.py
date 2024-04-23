from customtkinter import *
from PIL import Image

import cv2
import typing
import numpy as np
import language_tool_python
from PIL import Image
from tkinter.ttk import *
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer

class TrangChu(CTkFrame):
    def __init__(self, master, **kwargs):
        self.master = master
        super().__init__(master, **kwargs)
        self.pack_propagate(0)
        self.configure(width=1080, height=720, fg_color="#ffffff", corner_radius=0)
        self.initModel()
        self.pdfTextArray=[]
        self.final_predict =""

        global uploadBtn 
        uploadBtn = CTkButton(
            master=self, 
            width=150, 
            height=45, 
            fg_color="#6BD425", 
            hover_color="#618B25",
            text="Chọn ảnh",
            font=("Arial Bold",12),
            text_color="#FFFFFF",
            command=self.upLoadBtn_command
            )
        uploadBtn.place(x=20, y=20)

        clearImgBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#f0f7ee",
            hover_color="#8c8b8b",
            font=("Arial Bold", 12),
            text="Clear Image",
            text_color="#000000",
            border_color="#cccccc",
            border_width=1,
            corner_radius=0,
            command=self.clearImgBtn_command
        )
        clearImgBtn.place(x=20,y=90)

        clearTextBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#f0f7ee",
            hover_color="#8c8b8b",
            font=("Arial Bold", 12),
            text="Clear text",
            text_color="#000000",
            border_color="#cccccc",
            border_width=1,
            corner_radius=0,
            command=self.clearTextBtn_command
        )
        clearTextBtn.place(x=900,y=90)

        global imageLabel
        imageLabel = CTkLabel(
            master=self,
            font=('Arial Bold', 14),
            anchor="center",
            width=540,
            height=580,
            text="Click on 'Choose file to upload' to put image here"
        )
        imageLabel.place(x=0, y=140)
        self.current_image_path = ""

        global resultText
        resultText = CTkTextbox(
            master=self,
            width=540,
            height=580,
            font=("Arial Bold", 14),
            border_width=1,
            text_color="#000000",
        )
        resultText.insert("0.0", "Your text goes here")
        resultText.configure(state="disabled")
        resultText.place(x=540, y=140)

        global catAnhBtn
        catAnhBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            font=("Arial Bold", 12),
            text="Cắt ảnh",
            text_color="#ffffff",
            fg_color="#6bd425",
            hover_color="#618B25",
            command=self.catAnhBtn_command
        )

        global lamXamAnhBtn
        lamXamAnhBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#6bd425",
            hover_color="#618B25",
            text="Xem từng bước cắt ảnh \n(bước 1 làm xám ảnh)",
            text_color="#ffffff",
            font=("Arial Bold", 11),
            command=self.lamXamAnhBtn_command
        )

        global toChuBtn
        toChuBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#6bd425",
            hover_color="#618B25",
            text="Bước 2: Tô Chữ",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.toChuBtn_command
        )

        global dongKhungBtn
        dongKhungBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#6bd425",
            hover_color="#618B25",
            text="Bước 3: Đóng khung",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.dongKhungBtn_command
        )

        global duDoanChuBtn
        duDoanChuBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#5bc0be",
            hover_color="#1F7A8C",
            text="Dự đoán chữ",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.duDoanChuBtn_command
        )

        global duDoanTungCauBtn
        duDoanTungCauBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#6bd425",
            hover_color="#618B25",
            text="Dự đoán từng câu",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.duDoanTungcauBtn_command
        )

        global duDoanToanBoBtn
        duDoanToanBoBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#5bc0be",
            hover_color="#1F7A8C",
            text="Dự Đoán Toàn Bộ",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.duDoanToanBoBtn_command
        )

    def ChangeText(self, changeText):
        resultText.configure(state="normal")
        resultText.delete(1.0,END)
        resultText.insert(END,changeText)
        resultText.configure(state="disabled")

    def showImg(self,inputImage):
        imageLabel.configure(text="")
        displayImg = ImageResizer.resize_maintaining_aspect_ratio(
            inputImage,
            imageLabel.cget("width"),
            imageLabel.cget("height")
        )
        im = Image.fromarray(displayImg)
        # imtk = ImageTk.PhotoImage(im)
        imCtk = CTkImage(
            light_image=im,
            size=(imageLabel.cget("width"), imageLabel.cget("height"))
        )
        imageLabel.configure(image=imCtk)

    def resetDisplay(self):
        catAnhBtn.place_forget()
        duDoanChuBtn.place_forget()
        lamXamAnhBtn.place_forget()
        toChuBtn.place_forget()
        dongKhungBtn.place_forget()
        duDoanTungCauBtn.place_forget()
        duDoanToanBoBtn.place_forget()
        resultText.configure(state="disabled")


    def upLoadBtn_command(self):
        print("Choose file to upload")
        self.UploadAction()
        uploadBtn.pack_forget()
        catAnhBtn.place(x=20,y=20)

    def clearTextBtn_command(self):
        print("clear Text")
        output = resultText.get(1.0,END)
        if(output != "Your text goes here\n"):
            self.ChangeText("")

    def clearImgBtn_command(self):
        print("clear Image")
        img = Image.open("img/image.png")
        nullImage = CTkImage(light_image=img)
        imageLabel.configure(image=nullImage)
        imageLabel.configure(text="Click on 'Choose file to upload' to put image here")
        # Ẩn hết nút
        self.resetDisplay()


    def catAnhBtn_command(self):
        print("Cắt ảnh")
        self.cropImageAction()
        lamXamAnhBtn.place(x=20,y=20)
        duDoanChuBtn.place(x=200,y=20)

    def lamXamAnhBtn_command(self):
        print("Cắt ảnh b1")
        self.imgGrayAction()
        toChuBtn.place(x=20,y=20)

    def toChuBtn_command(self):
        print("Tô chữ")
        self.colorTextAction()
        toChuBtn.pack_forget()
        dongKhungBtn.place(x=20,y=20)

    def dongKhungBtn_command(self):
        print("Đóng khung")
        dongKhungBtn.configure(state="disabled")
        self.boxesImgAction()
        dongKhungBtn.place_forget()
        toChuBtn.place_forget()

    def duDoanChuBtn_command(self):
        print("dự đoán")
        self.final_predict = ""
        duDoanTungCauBtn.place(x=20,y=20)
        duDoanToanBoBtn.place(x=200,y=20)
        self.showImg(boxImg)

    def duDoanTungcauBtn_command(self):
        print("Dự đoán từng câu")
        self.PredictLineAction()

    def duDoanToanBoBtn_command(self):
        print("Dự đoán toàn bộ")
        self.PredictAllAction()

    def UploadAction(self, event=None):
        image_path = filedialog.askopenfilename()
        self.current_image_path = image_path
        print(self.current_image_path)
        global img
        img = cv2.imread(image_path)
        self.showImg(img)

    def cropImageAction(self):
        global imgList
        imgList = self.cropSentence()

    def imgGrayAction(self):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        global thresh2
        ret, thresh2 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        self.showImg(thresh2)

    def colorTextAction(self):
        global mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,2))
        mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)
        self.showImg(mask)

    def waithere(self, milisecond=300):
        var = IntVar()
        self.master.after(milisecond, var.set, 1)
        print("Waiting...")
        self.master.wait_variable(var)

    def boxesImgAction(self):
        bboxes = []
        bboxes_img = img.copy()
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0,0,255), 2)
            self.waithere()
            self.showImg(bboxes_img)
            # app.showImg(bboxes_img)
            # cv2.imshow("",bboxes_img)
            # cv2.waitKey(0)
            bboxes.append((x,y,w,h))
        self.showImg(bboxes_img)

    def cropSentence(self):
        self.imgGrayAction()
        self.colorTextAction()
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
        self.showImg(bboxes_img)
        boxImg = bboxes_img
        return imgList
    
    def PredictLineAction(self):
        if(len(imgList) <= 0):
            self.duDoanToanBoBtn_command()
            return
        currImg = imgList.pop()
        self.showImg(currImg)
        self.ChangeText("Waiting...")
        self.waithere(100)
        predictText = self.model.predict(currImg)
        self.ChangeText(predictText)
        self.pdfTextArray.append(predictText)
        self.final_predict = self.final_predict + " " + predictText

    def PredictAllAction(self):
        self.ChangeText("waiting...")
        self.waithere(100)
        while len(imgList) >0:
            currImg = imgList.pop()
            predictText = self.model.predict(currImg)
            self.final_predict = self.final_predict + " " + predictText
            self.pdfTextArray.append(predictText)
        ShowResult = wordSpliter(self.final_predict)
        self.showImg(boxImg)
        self.ChangeText("\n".join(ShowResult))
        print(self.final_predict)

    def initModel(self):
        import pandas as pd
        from tqdm import tqdm
        from mltu.configs import BaseModelConfigs
        configs = BaseModelConfigs.load("Models/202301131202/configs.yaml")
        self.model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

def wordSpliter(str,n=10):
    pieces = str.split()
    return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))

def error_correcting(text):
    tool = language_tool_python.LanguageTool('en-US')
    datasets = tool.correct(text)
    return datasets

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