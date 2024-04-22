from customtkinter import *
from tkinter import messagebox as mb
from CTkTable import CTkTable
from PIL import Image

import cv2
import typing
import numpy as np
import language_tool_python
from PIL import Image
from fpdf import FPDF
from pyaspeller import YandexSpeller
from tkinter.ttk import *
import csv
from datetime import datetime
from pathlib import Path

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

class TrangChu(CTkFrame):
    def __init__(self, master, **kwargs):
        self.master = master
        super().__init__(master, **kwargs)
        self.pack_propagate(0)
        self.configure(width=1080, height=720, fg_color="#ffffff", corner_radius=0)
        self.initModel()

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

        global savePDFBtn
        savePDFBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#5bc0be",
            hover_color="#1F7A8C",
            font=("Arial Bold", 12),
            text="Save As Pdf",
            text_color="#ffffff",
            command=self.savePDFBtn_command
        )

        global imageLabel
        imageLabel = CTkLabel(
            master=self,
            font=('Arial Bold', 14),
            anchor="center",
            width=541,
            height=580,
            text="Click on 'Choose file to upload' to put image here"
        )
        imageLabel.place(x=0, y=140)
        self.current_image_path = ""

        global resultText
        resultText = CTkTextbox(
            master=self,
            width=540,
            height=570,
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

        global showErrorBtn
        showErrorBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#6bd425",
            hover_color="#618B25",
            text="Hiển Thị lỗi",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.showErorrBtn_command
        )

        global fixErrorBtn
        fixErrorBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#5bc0be",
            hover_color="#1F7A8C",
            text="Sửa Lỗi",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.fixErrorBtn_command
        )

        global editTextBtn
        editTextBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#6bd425",
            hover_color="#618B25",
            text="Tự chỉnh sửa thêm",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.editTextBtn_command
        )

        global saveEditedTextBtn
        saveEditedTextBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#6bd425",
            hover_color="#618B25",
            text="Lưu",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.saveEditedTextBtn_command
        )

        global cancelEditedTextBtn
        cancelEditedTextBtn = CTkButton(
            master=self,
            width=150,
            height=45,
            fg_color="#e9190f",
            hover_color="#FB3640",
            text="Hủy",
            text_color="#ffffff",
            font=("Arial Bold", 14),
            command=self.cancelEditedTextBtn_command
        )

        addImageBtn = CTkButton(
            master=self,
            width=220,
            height=45,
            fg_color="#cccccc",
            hover_color="#000000",
            text="Add To Important Images",
            text_color="#fffefe",
            font=("Arial Bold", 14),
            command=self.addImageBtn_command
        )
        addImageBtn.place(x=600, y=20)

        allImagesBtn = CTkButton(
            master=self,
            width=200,
            height=45,
            fg_color="#cccccc",
            hover_color="#000000",
            text="All Important Images",
            text_color="#fffefe",
            font=("Arial Bold", 14),
            command=self.show_all_images
        )
        allImagesBtn.place(x=380, y=20)

        self.image_path = None  # Khởi tạo image_path là None
        self.image_selected=False
        self.image_list = []  # Danh sách để lưu trữ các đường dẫn của các ảnh đã chọn
        self.flag=False


        # ===================
    
        
    def show_all_images(self):
        if self.image_list == []:        
            mb.showinfo("Empty!","No image!")
            return
        else:
            def upload_image(event,image_path,selected_image=True):
                # image_path = event.widget.image_path
                self.UploadAction(event,image_path,selected_image)
                self.image_selected=True
            
            # Tạo cửa sổ mới để hiển thị ảnh
            image_window = CTkToplevel(
                self.master
            )
            image_window.title("All Images")

            # # Tạo một Canvas để chứa ảnh và kết nối nó với thanh Scrollbar
            # canvas = tk.Canvas(image_window)
            
            # canvas.pack(side="left", fill="both", expand=True)

            # # Tạo thanh Scrollbar
            # scrollbar = tk.Scrollbar(image_window, orient="vertical", command=canvas.yview)
            # scrollbar.pack(side="right", fill="y")

            # # Liên kết thanh Scrollbar với Canvas
            # canvas.configure(yscrollcommand=scrollbar.set)

            # Tạo một frame để chứa các widget Label hiển thị ảnh
            image_frame = CTkScrollableFrame(
                master=image_window
            )
            image_frame.pack()
            # canvas.create_window((0, 0), window=image_frame, anchor="nw")
            for filepath in self.image_list:
                try:
                    # Mở ảnh bằng thư viện PIL
                    img = Image.open(filepath)
                    # Resize ảnh nếu cần thiết
                    # img.thumbnail((350, 350))
                    # Chuyển đổi ảnh sang định dạng mà tkinter có thể hiển thị
                    # img = ImageTk.PhotoImage(img)
                    # Tạo một widget Label để hiển thị ảnh
                    img_label = CTkLabel(
                        master=image_frame,
                        text="",
                        image=CTkImage(light_image=img)
                    )
                    img_label.image = img  # Giữ tham chiếu đến ảnh để tránh bị thu gom bởi Python
                    img_label.image_path = filepath  # Đặt đường dẫn của ảnh cho thuộc tính image_path
                    img_label.pack(padx=5, pady=5)

                    # Gắn sự kiện chuột click vào label để upload ảnh
                    img_label.bind("<Button-1>", lambda event, path=filepath: upload_image(event,path) & self.image_path==path)
                    
                    # Thêm sự kiện chuột cho hover
                    # img_label.bind("<Enter>", lambda event, label=img_label: label.config(borderwidth=2, relief="solid"))
                    # img_label.bind("<Leave>", lambda event, label=img_label: label.config(borderwidth=0, relief="flat"))
                except Exception as e:
                    print(f"Error loading image: {e}")
        print(self.image_list)

        # Cấu hình Canvas để lấy kích thước của frame chứa ảnh
        image_frame.update_idletasks()
        frame_width = image_frame.winfo_reqwidth()
        frame_height = image_frame.winfo_reqheight()

        # Đặt kích thước của Canvas và kích thước khu vực cuộn của nó
        # canvas.config(scrollregion=(0, 0, frame_width, frame_height))

        # Kết nối sự kiện lăn chuột với hàm xử lý
        # canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

    # =========================
    def addImageBtn_command(self):
        if self.image_selected:  # Kiểm tra xem ảnh đã được chọn chưa
            result = mb.askquestion('Add To Important Image', 'Do you really want to Add To List?')
            if result == 'yes':
                for filePath in self.image_list:
                    if self.image_path == filePath:
                        mb.showinfo("Existed!","Image existed!")
                        self.flag=True
                        break
                if self.flag==True:
                    print("Image Existed")
                    return
                else:
                    self.image_list.append(self.image_path)
                    # app.show_all_images(self)
                    print("Them thanh cong:", self.image_path)  
            else:
                print("Operation cancelled.")
        else:
            print("No image selected.")


    def ChangeText(self, changeText):
        resultText.configure(state="normal")
        resultText.delete(1.0,END)
        resultText.insert(END,changeText)
        resultText.configure(state="disabled")

    def showImg(self,inputImage):
        imageLabel.configure(text="")
        displayImg = ImageResizer.resize_maintaining_aspect_ratio(
            inputImage,
            imageLabel.winfo_width(),
            imageLabel.winfo_height()
        )
        im = Image.fromarray(displayImg)
        # imtk = ImageTk.PhotoImage(im)
        imCtk = CTkImage(
            light_image=im,
            dark_image=im,
            size=(imageLabel.winfo_width(), imageLabel.winfo_height())
        )
        imageLabel.configure(image=imCtk)
        imageLabel.image = imCtk

    def resetDisplay(self):
        catAnhBtn.place_forget()
        duDoanChuBtn.place_forget()
        lamXamAnhBtn.place_forget()
        toChuBtn.place_forget()
        dongKhungBtn.place_forget()
        duDoanTungCauBtn.place_forget()
        duDoanToanBoBtn.place_forget()
        fixErrorBtn.place_forget()
        showErrorBtn.place_forget()
        editTextBtn.place_forget()
        saveEditedTextBtn.place_forget()
        cancelEditedTextBtn.place_forget()
        savePDFBtn.place_forget()
        resultText.configure(state="disabled")

    def saveAsPdf(self, event=None):
        
        pdf = FPDF()
    
            #Add a page
        pdf.add_page()

            #set style and size of font 
            #that you want in the pdf
        pdf.set_font("Arial", size = 15)
            
            # pdf.cell(200,35,txt=output_data,ln=10,align='J')
        for x in self.pdfTextArray:
            pdf.cell(200, 10, txt = x, ln = 10, align = 'J') 
                
        pdf.output("output.pdf")     
        mb.showinfo("Succeed!","Save as PDF Successfully!")


    def upLoadBtn_command(self):
        print("Choose file to upload")
        # self.ChangeText("Loading...")
        if not self.image_path:
            # Nếu chưa, mở hộp thoại dialog để chọn ảnh
            self.UploadAction(selected_image=False)
        else:
            # Nếu đã có image_path, chỉ cần truyền selected_image=True cho UploadAction
            self.UploadAction(image_path=self.image_path, selected_image=True)
        uploadBtn.pack_forget()
        catAnhBtn.place(x=20,y=20)

    def clearTextBtn_command(self):
        print("clear Text")
        output = resultText.get(1.0,END)
        if(output != "Your text goes here\n"):
            self.ChangeText("")

    def clearImgBtn_command(self):
        print("clear Image")
        self.image_selected=False
        self.image_path=None
        img = Image.open("img/image.png")
        nullImage = CTkImage(light_image=img)
        imageLabel.configure(image=nullImage)
        imageLabel.configure(text="Click on 'Choose file to upload' to put image here")
        # Ẩn hết nút
        self.resetDisplay()


    def savePDFBtn_command(self):
        print("Save as pdf")
        self.saveAsPdf()


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
        showErrorBtn.place(x=20,y=20)
        fixErrorBtn.place(x=200,y=20)

    def showErorrBtn_command(self):
        print("Hiển thị lỗi")
        self.showErrorAction()

    def fixErrorBtn_command(self):
        print("Sửa Lỗi")
        self.pdfTextArray = []
        self.result_txt =""
        self.fixErrorAction()
        editTextBtn.place(x=20,y=20)
        duDoanChuBtn.place_forget()
        fixErrorBtn.place_forget()
        duDoanToanBoBtn.place_forget()

    def editTextBtn_command(self):
        print("Người dùng tự sửa thêm")
        saveEditedTextBtn.place(x=20,y=20)
        cancelEditedTextBtn.place(x=200,y=20)
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

    

    # def UploadAction(self, event=None):
    #     image_path = filedialog.askopenfilename()
    #     self.current_image_path = image_path
    #     print(self.current_image_path)
    #     global img
    #     img = cv2.imread(image_path)
    #     self.showImg(img)

    def UploadAction(self, event=None, image_path=None, selected_image=False):
        print(selected_image)
        print("UploadAction ", image_path)
        global img
        if selected_image is True:
            if image_path:
                img = cv2.imread(image_path)
                # Thực hiện các thao tác tiếp theo với ảnh đã chọn từ danh sách
                self.showImg(img)
            else:
                print("Không có hình trong danh sách.")
        else:
            # Nếu không có ảnh được chọn từ danh sách, mở hộp thoại để chọn ảnh
            if image_path is None:
                image_path = filedialog.askopenfilename()
                self.image_selected= True
            if image_path:
                self.image_path=image_path
                img = cv2.imread(image_path)
                # Thực hiện các thao tác tiếp theo với ảnh đã chọn từ hộp thoại
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
        self.final_predict = self.final_predict + " " + predictText

    def PredictAllAction(self):
        self.ChangeText("waiting...")
        self.waithere(100)
        while len(imgList) >0:
            currImg = imgList.pop()
            predictText = self.model.predict(currImg)
            self.final_predict = self.final_predict + " " + predictText
        ShowResult = wordSpliter(self.final_predict)
        self.showImg(boxImg)
        self.ChangeText("\n".join(ShowResult))
        print(self.final_predict)

    def showErrorAction(self):
        self.ChangeText("waiting...")
        self.waithere(100)
        print("start checking")
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(self.final_predict)

        resultStr =""
        for i in matches:
            resultStr += '\n'+ i.message + '\n' + i.context
            # resultStr += '\n{}\n{}\n'.format(
            # i.context, ' ' * (i.offsetInContext)+ '^' * i.errorLength
            # )
        self.ChangeText(resultStr)

    def fixErrorAction(self):
        self.ChangeText("waiting...")
        self.waithere(100)
        print("start fixing")
        # output_text = error_correct_pyspeller(app.final_predict)
            # print(output_text)

        output_data = error_correcting(self.final_predict)
        result = wordSpliter(output_data)
        self.pdfTextArray = wordSpliter(output_data)
        self.result_txt = "\n".join(result)
        self.ChangeText(self.result_txt)
        savePDFBtn.place(x=200,y=20)
        file_name, file_extension = os.path.splitext(self.current_image_path)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H--%M--%S")
        history_image_name = f'{dt_string}{file_extension}'
        import shutil
        shutil.copyfile(self.current_image_path, f'history/images/{history_image_name}')
        with open('history/history.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='|', quotechar='"', quoting=csv.QUOTE_ALL)
            csvwriter.writerow([history_image_name, self.result_txt])

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