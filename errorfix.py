import language_tool_python
import csv
from autocorrect import Speller
from datetime import datetime
from customtkinter import *
from trangchu import wordSpliter
import shutil

class errorFix(CTkFrame):
    def __init__(self, master,TrangChu, **kwargs):
        
        self.master = master
        super().__init__(master, **kwargs)
        self.pack_propagate(0)
        self.configure(width=1080, height=720, fg_color="#ffffff", corner_radius=0)

        global textArea
        textArea= CTkTextbox(
            master=self,
            width=950,
            height=400,
            bg_color="#FFFFFF",
            font=("Arial Bold",14),
            text_color="#000000"
        )
        textArea.insert(END,"Result Text")
        textArea.configure(state='disabled')
        textArea.place(x=80,y=70)

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
            command=lambda: self.showErorrBtn_command(TrangChu)
        )
        showErrorBtn.place(x=235,y=570)

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
            command=lambda: self.fixErrorBtn_command(TrangChu)
        )
        fixErrorBtn.place(x=750,y=570)

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
            command=lambda: self.editTextBtn_command(TrangChu)
        )
        editTextBtn.place(x=80,y=20)

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
            command=lambda: self.saveEditedTextBtn_command(TrangChu)
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
            command=lambda: self.cancelEditedTextBtn_command(TrangChu)
        )

    def showErorrBtn_command(self,TrangChu):
        print("Kiểm tra lỗi")
        self.showErrorAction(TrangChu)

    def fixErrorBtn_command(self,TrangChu):
        print("sửa lỗi")
        self.fixErrorAction(TrangChu)

    def changeText(self,text):
        textArea.configure(state='normal')
        textArea.delete(1.0,END)
        textArea.insert(END,text)
        textArea.configure(state='disabled')

    def setDefault(self,TrangChu):
        textArea.configure(state='normal')
        textArea.delete(1.0,END)
        resultText = TrangChu.final_predict
        textArea.insert(END,"\n".join(wordSpliter(resultText)))
        textArea.configure(state='disabled')
        saveEditedTextBtn.place_forget()
        cancelEditedTextBtn.place_forget()

    def waithere(self, milisecond=300):
        var = IntVar()
        self.master.after(milisecond, var.set, 1)
        print("Waiting...")
        self.master.wait_variable(var)

    def fixErrorAction(self,TrangChu):
        self.changeText("waiting...")
        self.waithere(100)
        print("start fixing")
        spell = Speller()
        output_text = spell(TrangChu.final_predict)
            # print(output_text)

        output_data = error_correcting(output_text)
        TrangChu.final_predict = output_data
        result = wordSpliter(output_data)
        resultArray = wordSpliter(output_data)
        TrangChu.pdfTextArray = []
        for i in resultArray:
            TrangChu.pdfTextArray.append(i)
        self.result_txt = "\n".join(result)
        self.changeText(self.result_txt)
        TrangChu.ChangeText(self.result_txt)
        if(os.path.isdir('history/images')):
            self.createHistory(TrangChu)
        if not (os.path.isdir('history/images')):
            os.makedirs('history/images')
            self.createHistory(TrangChu)

    def createHistory(self, TrangChu):
        file_name, file_extension = os.path.splitext(TrangChu.current_image_path)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H--%M--%S")
        history_image_name = f'{dt_string}{file_extension}'
        shutil.copyfile(TrangChu.current_image_path, f'history/images/{history_image_name}')
        with open('history/history.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='|', quotechar='"', quoting=csv.QUOTE_ALL)
            csvwriter.writerow([history_image_name, self.result_txt])

    def showErrorAction(self,TrangChu):
        self.changeText("waiting...")
        self.waithere(100)
        print("start checking")
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(TrangChu.final_predict)

        resultStr =""
        for i in matches:
            resultStr += '\n'+ i.message + '\n' + i.context
            # resultStr += '\n{}\n{}\n'.format(
            # i.context, ' ' * (i.offsetInContext)+ '^' * i.errorLength
            # )
        self.changeText(resultStr)

    def editTextBtn_command(self,TrangChu):
        print("Người dùng tự sửa thêm")
        textArea.configure(state='normal')
        textArea.delete(1.0,END)
        resultText = TrangChu.final_predict
        textArea.insert(END,"\n".join(wordSpliter(resultText)))
        textArea.configure(state='disabled')
        saveEditedTextBtn.place(x=80,y=20)
        cancelEditedTextBtn.place(x=280,y=20)
        textArea.configure(state="normal")

    def saveEditedTextBtn_command(self,TrangChu):
        print("Lưu")
        newOutput = textArea.get(1.0,END)
        outputArr = newOutput.split('\n')
        TrangChu.pdfTextArray = outputArr
        TrangChu.final_predict = newOutput
        TrangChu.ChangeText(newOutput)
        saveEditedTextBtn.place_forget()
        cancelEditedTextBtn.place_forget()
        textArea.configure(state="disabled")

    def cancelEditedTextBtn_command(self,TrangChu):
        print("Hủy")
        self.setDefault(TrangChu)

def error_correcting(text):
    tool = language_tool_python.LanguageTool('en-US')
    datasets = tool.correct(text)
    return datasets