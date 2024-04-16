# inferenceModel.py
import cv2
import typing
import asyncio
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
        GMessage_172=tk.Message(root)
        ft = tkFont.Font(family='Times',size=10)
        GMessage_172["font"] = ft
        GMessage_172["fg"] = "#333333"
        GMessage_172["justify"] = "center"
        GMessage_172["text"] = "Title"
        GMessage_172.place(x=540,y=140,width=540,height=570)

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

    def ChangeText(self,changeText):
        GMessage_172.configure(text=changeText)

    def showImg(self,inputImage):
        displayImg = ImageResizer.resize_maintaining_aspect_ratio(inputImage,1078,249)
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
        GMessage_172.configure(text="")
        # Ẩn hết nút
        GButton_788.place_forget()
        GButton_63.place_forget()
        GButton_656.place_forget()
        GButton_29.place_forget()
        GButton_894.place_forget()


    def GButton_470_command(self):
        print("clear Image")
        GLabel_277.configure(image="")
        # Ẩn hết nút
        GButton_788.place_forget()
        GButton_63.place_forget()
        GButton_656.place_forget()
        GButton_29.place_forget()
        GButton_894.place_forget()


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
        predictAction()
        GButton_63.place_forget()

    def GButton_29_command(self):
        print("Tô chữ")
        colorTextAction()
        GButton_894.place(x=20,y=20,width=150,height=45)

    def GButton_894_command(self):
        print("Đóng khung")
        asyncio.run(boxesImgAction())
        
        

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

async def boxesImgAction():
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # imgList = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0,0,255), 2)
        await asyncio.sleep(0.3)
        app.showImg(bboxes_img)
        # cv2.imshow("",bboxes_img)
        # cv2.waitKey(0)
        bboxes.append((x,y,w,h))

def cropSentence():
    imgGrayAction()
    colorTextAction()
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
    return imgList

def PredictionFormCropImg(Imglist):
    prediction = []
    for i in Imglist:
        prediction_text = model.predict(i)
        prediction.append(prediction_text)
    return prediction

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

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/202301131202/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

# df = pd.read_csv("Models/04_sentence_recognition/202301131202/val.csv").values.tolist()

    # accum_cer, accum_wer = [], []
    # # for image_path, label in tqdm(df):
    # image = cv2.imread(image_path)

    # prediction_text = model.predict(image)

    # print("Image: ", image_path)

    # print("Prediction: ", prediction_text)

    # cv2.imshow(prediction_text, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # test()
    
    # root = Tk()
    
    # root.title("Hand-writing Recognize")
    # root.geometry('1080x720')
    
    # resultText = Text(root, height=22,width=80)
    # resultText.configure(state=DISABLED, bg="white",fg="black",font=("Comic Sans MS",16))
    # # resultText.grid(row=1, column=0, sticky=EW)
    # resultText.pack(side=BOTTOM)
    
    # # create a scrollbar widget and set its command to the text widget
    # scrollbar = Scrollbar(root, orient='vertical', command=resultText.yview)
    # # scrollbar.grid(row=1, column=1, sticky=NS)

    # #  communicate back to the scrollbar
    # resultText['yscrollcommand'] = scrollbar.set
    
    # # top_left = Frame(root, bg='black', width=200, height=200)
    # # top_left.grid(row=0)
    
    # # lb = Label(root,text= "")
    # # lb.grid(column=0,row=1)
    
    # # lbImg = Label(root, image= "")
    # # lbImg.grid(column=0,row=1)
    
    # # lbImg = Label(top_left, image= "")
    # # lbImg.pack()
    
    def UploadAction(event=None):
        image_path = filedialog.askopenfilename()
        # print("Selected: ", filename)

        # img = crop_text(image_path)
        global img
        img = cv2.imread(image_path)
        # resizedImg = ImageResizer.resize_maintaining_aspect_ratio(img,1091,249)
        app.showImg(img)

    def cropImageAction():
        global imgList
        imgList = cropSentence()
        # prediction = cropSentence(img)
        
        # showImg = ImageTk.PhotoImage(Image.open(image_path))
    
    def predictAction():
        prediction = PredictionFormCropImg(imgList)
        final_prediction = " "
        while prediction:
            if len(prediction) == 0:
                break
            final_prediction += prediction.pop() + ' '
        # print(final_prediction)
        # lb.configure(text = final_prediction)

        # tool = language_tool_python.LanguageTool('en-US')
        # matches = tool.check(final_prediction)
        # print(len(matches))
        # for i in matches:
        #     print(i)
        output_text = error_correct_pyspeller(final_prediction)
        # print(output_text)

        output_data = error_correcting(output_text)
        # print(output_data)
        # matches = tool.check(output_text)
        # print(len(matches))
        # for i in matches:
        #     print(i)        
        resultTextInputArray = wordSpliter(output_data)
        # resultText.configure(state=NORMAL)
        # resultText.insert(END,"\n".join(resultTextInputArray))
        # resultText.configure(state=DISABLED)
        app.ChangeText("\n".join(resultTextInputArray))
        # lbImg.configure(image = showImg)
        global pdfTextArray
        
        pdfTextArray = wordSpliter(output_data)
    
    # def clearResultText(event=None):
    #     resultText.configure(state=NORMAL)
    #     resultText.delete(1.0,END)
    #     resultText.configure(state=DISABLED)
        
    # # Hàm lưu file pdf #   
    def saveAsPdf(event=None):
        
        pdf = FPDF()
  
         #Add a page
        pdf.add_page()

         #set style and size of font 
         #that you want in the pdf
        pdf.set_font("Arial", size = 15)
        
        # pdf.cell(200,35,txt=output_data,ln=10,align='J')
        for x in pdfTextArray:
            pdf.cell(200, 10, txt = x, ln = 10, align = 'J') 
            
        pdf.output("output.pdf")     
        mb.showinfo("success","Save Pdf Successfully")
        
    def showConfirmDialog(event=None):
        result = mb.askquestion('Exit Application', 'Do you really want to exit?') # mb = messagebox (import from tkinter)
        
        if(result == 'yes'):
            root.destroy()
        else:
            return

    # # Create style Object
  
 

 
    # # Changes will be reflected
    # # by the movement of mouse.
    # # click_btn= PhotoImage(file='D:\Study\Python\hand_writing\Images\click.png')
    

    # uploadBtn = Button(root, text="Choose file to upload",command=UploadAction)
    # saveAsPdfBtn = Button(root, text="Save as PDF",command=saveAsPdf)
    # clearResultTextBtn = Button(root, text="Clear Text",command=clearResultText)
    # closeAppBtn = Button(root, text="Quit Application",command=showConfirmDialog)
    
    # uploadBtn.pack(side=LEFT)
    # clearResultTextBtn.pack(side=LEFT)
    # saveAsPdfBtn.pack(side=LEFT)
    # closeAppBtn.pack(side=LEFT)
    
    
    # # menu = Menu(root)
    # # root.config(menu=menu)
    # # filemenu = Menu(menu)
    # # menu.add_cascade(label='File', menu=filemenu)
    # # filemenu.add_command(label="Choose a file to upload", command=UploadAction)
    
    # root.mainloop()
    
    # cv2.waitKey(0)

    root = tk.Tk()
    app = App(root)
    root.mainloop()