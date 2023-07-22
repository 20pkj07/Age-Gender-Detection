# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import cv2

# Loading the Model
model = load_model('Age_Gender_Detection.h5')


# Initializing the GUI
root = tk.Tk()
root.geometry('800x600')
root.minsize(600,600)
root.maxsize(1920,1080)
root.configure(background="skyblue")
root.title("Pankaj's Age Gender Detection")


# Initializing the Labels 
f1 = Frame(root, bg="grey", borderwidth=6, relief=SUNKEN)
f1.pack(side=TOP,fill=X)
f2 = Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f2.pack(side=LEFT,fill=Y)
f3 = Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f3.pack(side=RIGHT,fill=Y)
f4 = Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f4.pack(side=BOTTOM,fill=X)
f5 =Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f5.pack(side=TOP,pady=20,padx=20)
label0 = Label(f4,text="  " ,background="silver", font=('lucida', 30, "bold")).pack()
label01 = Label(f2,text="   " ,background="gray64", font=('lucida', 15, "bold")).pack()
label02= Label(f3,text="   " ,background="gray64", font=('lucida', 15, "bold")).pack()

label1 = Label(f5, background="#CDCDCD", font=('lucida', 15, "bold"))
label2 = Label(f5, background="#CDCDCD", font=('lucida', 15, 'bold'))
label3 = Label(f5, background="#CDCDCD", font=('lucida', 15, 'bold'))
sign_image = Label(root)
label1.configure(text='Gender   :  ',foreground="red")
label2.configure(text='Age   :  ',foreground="blue")
label3.configure(text='Age range :  ',foreground="green")

# Detect faces in the image
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]
    else:
        face_image =image
        
    img2 = cv2.resize(face_image,(64,64))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = np.array(img2)
    img2 = np.array([img2]) / 255
    return img2

# Defining Detect function which detects the age and gender of the person in an image using the model
def get_age(j):
    if j >= 0 and j <= 5:return "0-5"
    if j >5   and j <= 10:return "6-10"
    if j >10  and j <= 14:return "11-14"
    if j >15  and j <= 18:return "15-18"
    if j >18  and j <= 25:return "19-25"
    if j >25  and j <= 35:return "26-35"
    if j >35  and j <= 50:return "36-50"
    if j >50  and j <= 60:return "51-60"
    if j >60  and j <= 80:return "61-80"
    if j >80  and j <= 100:return "81-100"
    if j >100  and j <= 120:return "101-120"
    return "Unknown"


def Detect(file_path):
    global Label_packed
    image = Image.open(file_path)
    #image = cv2.imread(file_path)  
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.resize(image,(64,64))
    #image = np.array(image)
    #image = np.array([image]) / 255
    gen=["Male", "Female"]
    img = detect_faces(file_path)
    pred=model.predict(img)
    age=int(np.argmax(pred[1][0]))
    gender=int(np.round(pred[0][0]))
    print("Predicted Gender is "+ gen[gender])
    print("Predicted Age is "+ str(age))
    print("Predicted Age range is " + get_age(age))
    label1.configure(foreground="red", text=f"Gender :  {gen[gender]}")
    label2.configure(foreground="blue", text=f"Age  :  {age}")
    label3.configure(foreground="green", text=f"Age range :  {get_age(age)}")

# Defining Show_detect button function
def show_Detect_button(file_path):
    try:
        Detect_b = Button(f4, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
        Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        Detect_b.place(relx=0.55, rely=0.1)
        
    except:
        pass

# Defining Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename(filetypes =[('Image Files', '*.jpg')])
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((root.winfo_width() / 2.25), (root.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        label1.configure(text='Gender   :  ',foreground="red")
        label2.configure(text='Age   :  ',foreground="blue")
        label3.configure(text='Age range :  ',foreground="green")

        sign_image.configure(image=im)
        sign_image.image = im
        
        show_Detect_button(file_path)

    except:
        pass

upload = Button(f4, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=5)
upload.place(relx=0.3, rely=0.1)
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)
label3.pack(side="bottom", expand=True)
heading = Label(f1, text="Age and Gender Detector", pady=10, font=('arial', 20, "bold"),)
heading.configure(background="silver", foreground="black")
heading.pack(fill=X)


root.mainloop()
