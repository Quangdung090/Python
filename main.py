# inferenceModel.py
import cv2
import typing
import numpy as np
import language_tool_python
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from fpdf import FPDF
from pyaspeller import YandexSpeller
from tkinter.ttk import *



# from autocorrect import Speller
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

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

def cropSentence(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,2))
    mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    prediction = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0,0,255), 2)
        crop_img = img[y:y+h, x:x+w]
        prediction_text = model.predict(crop_img)
        cv2.putText(bboxes_img, prediction_text, (x, y -4), 0, 0.02*h, (30,15,252), ((w+h)//900)//3)
        prediction.append(prediction_text)

        # print("Prediction: ", prediction_text)
        # cv2.imshow(predictiont_tex,crop_img)
        # cv2.waitKey(0)
        bboxes.append((x,y,w,h))
    bboxes_img = ImageResizer.resize_maintaining_aspect_ratio(bboxes_img,1840,720)
    cv2.imshow("",bboxes_img)
    return prediction

def crop_text(img_path):
    img = cv2.imread(img_path)
  # Read in the image and convert to grayscale
    img = img[:-20, :-20]  # Perform pre-cropping
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 50).astype(np.uint8)  # To invert the text to white
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones(
    (2, 2), dtype=np.uint8))  # Perform noise filtering
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    # Crop the image - note we do this on the original image
    rect = img[y-2:y+h-4, x:x+w]
    return rect


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
    
    root = Tk()
    
    root.title("Hand-writing Recognize")
    root.geometry('1080x720')
    
    resultText = Text(root, height=20, width=50)
    resultText.configure(state=DISABLED, bg="white",fg="black",font=("Comic Sans MS",16))
    resultText.grid(row=1, column=0, sticky=EW)
    
    # create a scrollbar widget and set its command to the text widget
    scrollbar = Scrollbar(root, orient='vertical', command=resultText.yview)
    scrollbar.grid(row=1, column=1, sticky=NS)

    #  communicate back to the scrollbar
    resultText['yscrollcommand'] = scrollbar.set
    
    # top_left = Frame(root, bg='black', width=200, height=200)
    # top_left.grid(row=0)
    
    # lb = Label(root,text= "")
    # lb.grid(column=0,row=1)
    
    # lbImg = Label(root, image= "")
    # lbImg.grid(column=0,row=1)
    
    # lbImg = Label(top_left, image= "")
    # lbImg.pack()
    
    def UploadAction(event=None):
        image_path = filedialog.askopenfilename()
        # print("Selected: ", filename)

        # img = crop_text(image_path)
        img = cv2.imread(image_path)
        prediction = cropSentence(img)
        
        # showImg = ImageTk.PhotoImage(Image.open(image_path))
    
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
        testArray = wordSpliter(output_data)
        resultText.configure(state=NORMAL)
        resultText.insert(END,"\n".join(testArray))
        resultText.configure(state=DISABLED)
        # lbImg.configure(image = showImg)
        
        # print(testArray)
        
        pdf = FPDF()   
  
         #Add a page
        pdf.add_page()

         #set style and size of font 
         #that you want in the pdf
        pdf.set_font("Arial", size = 15)
        

        # pdf.cell(200,35,txt=output_data,ln=10,align='J')
        for x in testArray:
            pdf.cell(200, 10, txt = x, ln = 10, align = 'J') 
            
        pdf.output("output.pdf")  
        

    # Create style Object
    style = Style()
 
    style.configure('TButton', font =('calibri', 20, 'bold'),borderwidth = '4')
 
    # Changes will be reflected
    # by the movement of mouse.
    style.map('TButton', foreground = [('active', '!disabled', 'green')],background = [('active', 'black')])
    uploadBtn = Button(root, text="Choose file to upload", command=UploadAction)
    uploadBtn.grid(column=0,row=0,padx=400,pady=10)
    
    root.mainloop()
    
    cv2.waitKey(0)