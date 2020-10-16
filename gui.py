
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image





#load the model
model = tf.keras.models.load_model('model') 


def process(img):
    images=[]
    cv2.imwrite('5.png',np.float32(img)) #SAVE THE IMAGE
    col = Image.open('5.png') # READ THE IMAGE
    gray = col.convert('L') #CONVERT THE IMAGE
    bw = gray.point(lambda x: 0 if x<100 else 255, '1')
    bw.save('5.png')

    img = cv2.imread('5.png',cv2.IMREAD_GRAYSCALE) #READ THE IMAGE 
    img = cv2.bitwise_not(img)
    img_size = 28
    img = cv2.resize(img, (img_size,img_size))
    finalIMAGE = tf.keras.utils.normalize(img, axis = 1)
    images.append(finalIMAGE)
    npa = np.asarray(images, dtype=np.float32)
    npa = npa.reshape(npa.shape[0], 28, 28, 1)
    predictions = model.predict(npa)
    print(np.argmax(predictions[0]))
    print(predictions[0])
    return np.argmax(predictions[0]),max (predictions)




class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=200, height=200, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognize", command =         self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = process(im)

        self.label.configure(text= str(digit))
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=15
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black',width=0)
app = App()
mainloop()



    

    



