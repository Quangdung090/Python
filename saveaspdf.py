from customtkinter import *
from tkinter import messagebox as mb
from tkinter import messagebox as mb
import re
from fpdf import FPDF


class SavePdfGui(CTkFrame):
    def __init__(self, master, **kwargs):
        
        self.master = master
        super().__init__(master, **kwargs)
        self.pack_propagate(0)
        self.configure(width=1080, height=720, fg_color="#ffffff", corner_radius=0)
        
        global textArea
        textArea= CTkTextbox(
            master=self,
            width=930,
            height=380,
            bg_color="#FFFFFF",
            font=("Arial Bold",14),
            text_color="#000000"
        )
        textArea.insert(END,"Result Text")
        textArea.configure(state='disabled')
        textArea.place(x=80,y=70)

        inputLabel = CTkLabel(
            master=self,
            width=100,
            height=25,
            font=("Arial Bold",14),
            text_color="#000000",
            text="file name: "
        )
        inputLabel.place(x=490,y=470)

        self.filename = ""

        global inputBox
        inputBox = CTkEntry(
            master=self,
            width=220,
            height=45,
            textvariable= self.filename,
            font=("Arial Bold",14),
            placeholder_text="File Name"
        )
        inputBox.place(x=430,y=500)

        savePDFBtn = CTkButton(
            master=self,
            width=220,
            height=45,
            fg_color="#5bc0be",
            hover_color="#1F7A8C",
            font=("Arial Bold", 12),
            text="Save As Pdf",
            text_color="#ffffff",
            command=self.savePDFBtn_command
        )
        savePDFBtn.place(x=430,y=570)

    def savePDFBtn_command(self):
        print("Save as pdf")

        if(inputBox.get() == ""):
            return
        if(re.search("^[a-zA-Z]*$",inputBox.get()) == None):
            mb.showinfo("Error","Tên file chỉ có chữ không cách")
            return
        pdf = FPDF()
    
            #Add a page
        pdf.add_page()

            #set style and size of font 
            #that you want in the pdf
        pdf.set_font("Arial", size = 16)
            
            # pdf.cell(200,35,txt=output_data,ln=10,align='J')
        for x in self.textArray:
            pdf.cell(200, 10, txt = x, ln = 10, align = 'J') 
                
        pdf.output(inputBox.get() + ".pdf")     
        mb.showinfo("Succeed!","Save as PDF Successfully!")

    def setText(self,TrangChu):
        textArea.configure(state='normal')
        textArea.delete(1.0,END)
        if(len(TrangChu.pdfTextArray)>0):
            self.textArray = TrangChu.pdfTextArray
            resultArray = self.textArray
            textArea.insert(END,"\n".join(resultArray))
        else:
            textArea.insert(END,"Result Text")
        textArea.configure(state='disabled')
        
