import cv2
from tkinter import Label, Tk, Button, ttk
import tkinter as tkMe
from PIL import ImageTk, Image
import os
import numpy as np
import sys

win = Tk()
win.geometry("600x600")
# win.configure(background='brown')
filters = "RGB", "GRAY", "HSV", "FHSV", "HLS"
combo_box = ttk.Combobox(win, values=filters)
combo_box.current(0)
combo_box.place(x=50, y=500, width=100)

# TODO: Add output dimension
output_width = 300
output_height = 400


def change_filter(img: cv2.typing.MatLike):
    if combo_box.get() == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif combo_box.get() == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif combo_box.get() == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif combo_box.get() == "FHSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    elif combo_box.get() == "HLS":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return img


# TODO: edit image path
images_path = 'Palm/Source'
assert os.path.exists(images_path), f"Path not found: {images_path}"
myList = os.listdir(images_path)
ListSourceImage = [img for img in myList if img.endswith(('.JPG', '.jpg', '.png'))]

current_image: cv2.typing.MatLike = None
canvas_image: cv2.typing.MatLike = None
drawing_lines = None
current_filename: str = ""


def to_display(img: cv2.typing.MatLike, box_label: Label, x: int, y: int, w: int, h: int):
    global current_image, canvas_image, drawing_lines, current_filename
    img = cv2.resize(img, (w, h))
    current_image = img
    image = Image.fromarray(img)
    pic = ImageTk.PhotoImage(image)
    box_label.configure(image=pic)
    box_label.image = pic
    box_label.place(x=x, y=y)
    canvas_image = img.copy()  # Keep a copy for drawing  // np.zeros_like
    drawing_lines = np.zeros_like(canvas_image)  # Initialize a blank canvas for drawing lines
    # drawing_lines[1:300,1:400] = (0,0,0)
    # drawing_lines= np.zeros(canvas_image.shape[:2], np.uint8)
    # drawing_lines = cv2.rectangle(mask_face, (0,0), (w,h), (255,255,255), -1)
    current_filename = ListSourceImage[count]  # Keep track of the current image filename
    label_middle = tkMe.Label(win, text=current_filename)
    label_middle.place(relx=0.1, rely=0.1, anchor='center')
    win.mainloop()


def switch(i):
    global current_image
    img = cv2.imread(os.path.join(images_path, ListSourceImage[i]))
    # img = change_filter(img)
    # TODO: Add
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    to_display(img, label, 150, 20, 300, 400)
    # TODO: Edit
    # img = cv2.resize(img, (w, h))
    current_image = img


count = 0


def erase():
    print("Erase enabled")


def count_up():
    global count
    count += 1
    if count > len(ListSourceImage) - 1:
        count = 0
    switch(count)


def count_down():
    global count
    count -= 1
    if count < 0:
        count = len(ListSourceImage) - 1
    switch(count)


is_drawing = False
start_x = None
start_y = None


def start_drawing(event):
    global start_x, start_y
    if is_drawing:
        start_x, start_y = event.x, event.y


def draw(event):
    global start_x, start_y, canvas_image, drawing_lines
    if is_drawing and start_x is not None and start_y is not None:
        # White line drawing
        cv2.line(drawing_lines, (start_x, start_y), (event.x, event.y), (255, 255, 255), 3)
        start_x, start_y = event.x, event.y
        # TODO: Edit
        # combined_image = cv2.addWeighted(canvas_image, 0.5, drawing_lines, 1, 0)
        combined_image = cv2.addWeighted(canvas_image, 0.9, drawing_lines, 1, 0)
        img = Image.fromarray(combined_image)
        pic = ImageTk.PhotoImage(img)
        label.configure(image=pic)
        label.image = pic


def stop_drawing(event):
    global start_x, start_y
    if is_drawing:
        start_x, start_y = None, None


def toggle_drawing():
    global is_drawing
    is_drawing = not is_drawing
    print("Drawing enabled" if is_drawing else "Drawing disabled")


def save_drawing():
    global drawing_lines, current_filename, current_image
    if drawing_lines is not None:
        # Ensure the output directory exists
        output_dir = "Palm/output"
        input_dir = "Palm/input"  # current_image
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        # TODO: Fix
        # # Save the drawing lines only
        # Outfile_path = os.path.join(output_dir, current_filename)
        # cv2.imwrite(Outfile_path, drawing_lines)
        # print(f"Drawing mask saved to {Outfile_path}")
        #
        # Infile_path = os.path.join(input_dir, current_filename)
        # cv2.imwrite(Infile_path, current_image)
        # print(f"Drawing Input Image saved to {Infile_path}")

        # Convert the drawing lines to BGR if needed before saving

        if len(drawing_lines.shape) == 2 or drawing_lines.shape[2] == 1:
            drawing_lines_bgr = cv2.cvtColor(drawing_lines, cv2.COLOR_GRAY2BGR)
        else:
            drawing_lines_bgr = drawing_lines

        Outfile_path = os.path.join(output_dir, current_filename)
        cv2.imwrite(Outfile_path, drawing_lines_bgr)
        print(f"Drawing mask saved to {Outfile_path}")

        # Convert the current image back to BGR before saving
        current_image_bgr = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
        Infile_path = os.path.join(input_dir, current_filename)
        cv2.imwrite(Infile_path, current_image_bgr)
        print(f"Drawing Input Image saved to {Infile_path}")


def close_drawing():
    sys.exit()


button_pos = {'x': 360, 'y': 500}
padding_x = 50

label = Label(win, bg="black")
label.pack()
label.bind("<ButtonPress-1>", start_drawing)
label.bind("<B1-Motion>", draw)
label.bind("<ButtonRelease-1>", stop_drawing)

Button(win, text="◀", bg="gray", fg="white", command=count_down).place(x=230, y=500, width=40)
Button(win, text="▶", bg="gray", fg="white", command=count_up).place(x=290, y=500, width=40)
Button(win, text="ลบ", bg="gray", fg="white", command=erase).place(x=350, y=500, width=40)
Button(win, text="Draw", bg="gray", fg="white", command=toggle_drawing).place(x=button_pos['x'] + padding_x,
                                                                              y=button_pos['y'], width=40)
Button(win, text="Save", bg="gray", fg="white", command=save_drawing).place(x=button_pos['x'] + 2 * padding_x,
                                                                            y=button_pos['y'], width=40)
Button(win, text="Exit", bg="gray", fg="white", command=close_drawing).place(x=button_pos['x'] + 3 * padding_x,
                                                                             y=button_pos['y'], width=40)
if __name__ == "__main__":
    switch(0)  # Load the first image initially
    win.mainloop()
