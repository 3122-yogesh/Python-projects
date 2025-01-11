import cv2
import numpy as np
from tkinter import *
from tkinter import font,Label,ttk,filedialog
from PIL import Image, ImageTk
import os
SHOP_NAME = "COMPANY LOGO DETECTION"

window =Tk()
img = ""
result = ""
dataset_dir = os.path.join(os.getcwd(),"dataset")

window.title(SHOP_NAME)
win_width= window.winfo_screenwidth()    
half_width = win_width//2
win_height= window.winfo_screenheight()         
half_height = win_height//2      
window.geometry("%dx%d" % (half_width, half_height))
window.config(bg="light grey")


LOGO_DATA = {
    f'{dataset_dir}\\facebook.jpeg': 'FACEBOOK',
    f'{dataset_dir}\\google.png': 'GOOGLE',
    f'{dataset_dir}\\instagram.jpeg':'INSTAGRAM',
    # Add more logos and corresponding company names as needed
}

def detect_logo(target_path):
    # Read target image
    target = cv2.imread(target_path, 0)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Initialize variables to store best match information
    best_match_score = float('inf')
    best_match_company = None

    for logo_path, company_name in LOGO_DATA.items():
        # Read logo image
        logo = cv2.imread(logo_path, 0)

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(logo, None)
        kp2, des2 = orb.detectAndCompute(target, None)

        # Create matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Check if there are any matches
        if matches:
            # Calculate match score
            match_score = sum([match.distance for match in matches]) / len(matches)

            # Update best match information if needed
            if match_score < best_match_score:
                best_match_score = match_score
                best_match_company = company_name

    return best_match_company

def open_filechooser():
    global img
    filepath = filedialog.askopenfilename(
        initialdir=".",
        title="Choose Image",
        filetypes=[("Image File",'.jpg .png .jpeg')]
    )
    print(filepath)
    # Call the detect_logo function
    company_name = detect_logo(filepath)

    if company_name:
        print("The detected company is:", company_name)
        res = "The detected company is:" + company_name
    else:
        print("No matching logo found.")
    image = Image.open(filepath)
    image = image.resize((200, 200)) 
    img= ImageTk.PhotoImage(image)
    imgLabel.pack(side=TOP,pady=30)
    imgLabel.config(image=img)
    imgLabel.image=img
    resultLabel.config(text=res)
    resultLabel.pack()

# TITLE FRAME 
title_frame = Frame(master=window,bg="light grey")
title_frame.pack(side=TOP)

title = Label(master=title_frame,text=SHOP_NAME, fg="red",bg="light grey", font = ('Verdana', 20) )
title.pack()

image_path = f'{dataset_dir}\\image.png'
image = ImageTk.PhotoImage(Image.open(image_path).resize((100,50)))

imgLabel = Label(window,image=img)
imgLabel.pack_forget()

resultLabel = Label(window,text="",fg="green",font = ('Verdana', 12) )

button = Button(window,text="",image=image,command=open_filechooser,bg="light grey")
button.pack(side=TOP)


def on_closing():
    window.destroy()
    exit()


window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()

