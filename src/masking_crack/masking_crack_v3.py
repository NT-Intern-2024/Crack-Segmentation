import cv2
from tkinter import Label, Tk, Button, ttk
import tkinter as tkMe
from PIL import ImageTk, Image
import os
import numpy as np
import sys
import platform

win = Tk()
win.geometry("600x600")
filters = "RGB", "GRAY", "HSV", "FHSV", "HLS"
combo_box = ttk.Combobox(win, values=filters)
combo_box.current(0)
combo_box.place(x=50, y=500, width=100)

output_width = 300
output_height = 400

images_path = 'Palm/Source'
assert os.path.exists(images_path), f"Path not found: {images_path}"
myList = os.listdir(images_path)

ListSourceImage = sorted([img for img in myList if img.endswith(('.JPG', '.jpg', '.png'))])

current_image = None
canvas_image = None
drawing_lines = None
current_filename = ""
drawing_history = []  # Stack to keep track of drawing actions
redo_stack = []  # Stack to keep track of redo actions


def change_filter(img):
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


def to_display(img, box_label, x, y, w, h):
    global current_image, canvas_image, drawing_lines, current_filename
    img = cv2.resize(img, (w, h))
    current_image = img
    image = Image.fromarray(img)
    pic = ImageTk.PhotoImage(image)
    box_label.configure(image=pic)
    box_label.image = pic
    box_label.place(x=x, y=y)
    canvas_image = img.copy()
    drawing_lines = np.zeros_like(canvas_image)
    current_filename = ListSourceImage[count]
    label_middle = tkMe.Label(win, text=current_filename)
    label_middle.place(relx=0.1, rely=0.1, anchor='center')
    win.mainloop()


def switch(i):
    global current_image, drawing_history, redo_stack
    drawing_history.clear()  # Clear the drawing history stack
    redo_stack.clear()  # Clear the redo stack
    img = cv2.imread(os.path.join(images_path, ListSourceImage[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    to_display(img, label, 150, 20, 300, 400)
    current_image = img


count = 0


def erase():
    global drawing_lines, label
    drawing_lines = np.zeros_like(canvas_image)
    update_canvas()


def update_canvas():
    global canvas_image, drawing_lines, label
    combined_image = cv2.addWeighted(canvas_image, 0.9, drawing_lines, 1, 0)
    img = Image.fromarray(combined_image)
    pic = ImageTk.PhotoImage(img)
    label.configure(image=pic)
    label.image = pic


def undo(event=None):
    global drawing_history, drawing_lines, redo_stack
    if drawing_history:
        redo_stack.append(drawing_lines.copy())
        drawing_lines = drawing_history.pop()
        update_canvas()


def redo(event=None):
    global drawing_history, drawing_lines, redo_stack
    if redo_stack:
        drawing_history.append(drawing_lines.copy())
        drawing_lines = redo_stack.pop()
        update_canvas()


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
    global start_x, start_y, canvas_image, drawing_lines, drawing_history, redo_stack
    if is_drawing and start_x is not None and start_y is not None:
        drawing_history.append(drawing_lines.copy())  # Save the current state for undo
        redo_stack.clear()  # Clear the redo stack when a new action is made
        cv2.line(drawing_lines, (start_x, start_y), (event.x, event.y), (255, 255, 255), 3)
        start_x, start_y = event.x, event.y
        update_canvas()


def stop_drawing(event):
    global start_x, start_y
    if is_drawing:
        start_x, start_y = None, None


def toggle_drawing():
    global is_drawing, draw_button
    is_drawing = not is_drawing
    if is_drawing:
        draw_button.config(text="Stop", bg="red")
    else:
        draw_button.config(text="Draw", bg="gray")
    print("Drawing enabled" if is_drawing else "Drawing disabled")


def save_drawing():
    global drawing_lines, current_filename, current_image
    if drawing_lines is not None:
        output_dir = "Palm/output"
        input_dir = "Palm/input"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        Outfile_path = os.path.join(output_dir, current_filename)
        cv2.imwrite(Outfile_path, drawing_lines)
        print(f"Drawing mask saved to {Outfile_path}")
        Infile_path = os.path.join(input_dir, current_filename)
        cv2.imwrite(Infile_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
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

Button(win, text="◀", bg="gray", command=count_down).place(x=230, y=500, width=40)
Button(win, text="▶", bg="gray", command=count_up).place(x=290, y=500, width=40)
Button(win, text="ลบ", bg="gray", command=erase).place(x=350, y=500, width=40)

draw_button = Button(win, text="Draw", bg="gray", command=toggle_drawing)
draw_button.place(x=button_pos['x'] + padding_x, y=button_pos['y'], width=40)

Button(win, text="Save", bg="gray", command=save_drawing).place(x=button_pos['x'] + 2 * padding_x,
                                                                y=button_pos['y'], width=40)
Button(win, text="Exit", bg="gray", command=close_drawing).place(x=button_pos['x'] + 3 * padding_x,
                                                                 y=button_pos['y'], width=40)
# Check the platform to determine the key bindings for undo and redo
if platform.system() == 'Darwin':  # macOS
    undo_key = '<Command-z>'
    redo_key = '<Command-y>'
else:  # Windows or Linux
    undo_key = '<Control-z>'
    redo_key = '<Control-y>'

# Bind undo and redo functions to the appropriate keys
win.bind(undo_key, undo)
win.bind(redo_key, redo)

if __name__ == "__main__":
    switch(0)
    win.mainloop()
