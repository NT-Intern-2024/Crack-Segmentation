import cv2
from tkinter import Label, Tk, Button, ttk, Radiobutton, IntVar, Listbox, Scrollbar, messagebox
import tkinter as tk
from PIL import ImageTk, Image
import os
import numpy as np
import sys

images_path = "../../data/Palm/PalmAll"

mask_output_path = "../../output/dataset/"

# Initialize the main window
win = Tk()
win.geometry("800x600")  # Adjust window size to fit all components

# Output dimensions
mask_width = 300
mask_height = 400
GRadivar = 1  # Default thickness
Radivar = IntVar(value=GRadivar)

toolbar_pos_y = mask_height + 100

filters = "RGB", "GRAY", "HSV", "FHSV", "HLS"
combo_box = ttk.Combobox(win, values=filters)
combo_box.current(0)
combo_box.place(x=30, y=toolbar_pos_y, width=100)

# Action stack
undo_stack = []
redo_stack = []


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


assert os.path.exists(images_path), f"Path not found: {images_path}"
myList = os.listdir(images_path)
ListSourceImage = sorted([img for img in myList if img.endswith(('.JPG', '.jpg', '.png'))])

current_image: cv2.typing.MatLike = None
canvas_image: cv2.typing.MatLike = None
drawing_lines = None
current_filename: str = ""


def to_display(img: cv2.typing.MatLike, box_label: Label, x: int, y: int, w: int, h: int):
    global current_image, canvas_image, drawing_lines, current_filename, undo_stack, redo_stack
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
    label_middle = tk.Label(win, text=current_filename)
    label_middle.place(relx=0.5, rely=0.95, anchor='center')
    undo_stack.clear()  # Clear the undo stack on image change
    redo_stack.clear()  # Clear the redo stack on image change
    win.mainloop()


def switch(i):
    global current_image
    img = cv2.imread(os.path.join(images_path, ListSourceImage[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    to_display(img, label, 200, 20, mask_width, mask_height)
    current_image = img


def on_image_select(event):
    global count
    selected_index = image_listbox.curselection()
    if selected_index:
        count = selected_index[0]
        switch(count)


count = 0


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


def erase():
    global drawing_lines
    drawing_lines = np.zeros_like(canvas_image)
    combined_image = cv2.addWeighted(canvas_image, 0.9, drawing_lines, 1, 0)
    img = Image.fromarray(combined_image)
    pic = ImageTk.PhotoImage(img)
    label.configure(image=pic)
    label.image = pic
    print("Drawing erased")


is_drawing = False
start_x = None
start_y = None


def start_drawing(event):
    global start_x, start_y
    if is_drawing:
        start_x, start_y = event.x, event.y


def sel():
    global GRadivar
    GRadivar = Radivar.get()


def draw(event):
    global start_x, start_y, canvas_image, drawing_lines, GRadivar, undo_stack, redo_stack
    if is_drawing and start_x is not None and start_y is not None:
        undo_stack.append(drawing_lines.copy())
        redo_stack.clear()  # Clear the redo stack whenever a new drawing is made
        cv2.line(drawing_lines, (start_x, start_y), (event.x, event.y), (255, 255, 255), int(GRadivar))
        start_x, start_y = event.x, event.y
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
    draw_button.config(bg="green" if is_drawing else "gray")
    print("Drawing enabled" if is_drawing else "Drawing disabled")


def save_drawing():
    global drawing_lines, current_filename, current_image
    if drawing_lines is not None:
        output_dir = os.path.join(mask_output_path, "output")
        input_dir = os.path.join(mask_output_path, "input")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        if len(drawing_lines.shape) == 2 or drawing_lines.shape[2] == 1:
            drawing_lines_bgr = cv2.cvtColor(drawing_lines, cv2.COLOR_GRAY2BGR)
        else:
            drawing_lines_bgr = drawing_lines

        Outfile_path = os.path.join(output_dir, current_filename)
        drawing_lines_bgr = cv2.resize(drawing_lines_bgr, (300, 400))
        cv2.imwrite(Outfile_path, drawing_lines_bgr)
        print(f"Drawing mask saved to {Outfile_path}")

        current_image_bgr = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
        Infile_path = os.path.join(input_dir, current_filename)
        current_image_bgr = cv2.resize(current_image_bgr, (300, 400))
        cv2.imwrite(Infile_path, current_image_bgr)
        print(f"Drawing Input Image saved to {Infile_path}")


def close_drawing():
    sys.exit()


def undo():
    global drawing_lines, undo_stack, redo_stack
    if undo_stack:
        redo_stack.append(drawing_lines.copy())  # Push current state to redo stack
        drawing_lines = undo_stack.pop()  # Pop previous state from undo stack
        combined_image = cv2.addWeighted(canvas_image, 0.9, drawing_lines, 1, 0)
        img = Image.fromarray(combined_image)
        pic = ImageTk.PhotoImage(img)
        label.configure(image=pic)
        label.image = pic
    else:
        print("Undo stack is empty.")


def redo():
    global drawing_lines, undo_stack, redo_stack
    if redo_stack:
        undo_stack.append(drawing_lines.copy())
        drawing_lines = redo_stack.pop()
        combined_image = cv2.addWeighted(canvas_image, 0.9, drawing_lines, 1, 0)
        img = Image.fromarray(combined_image)
        pic = ImageTk.PhotoImage(img)
        label.configure(image=pic)
        label.image = pic
    else:
        print("Redo stack is empty.")


button_pos = {'x': 360, 'y': 500}
padding_x = 50

label = Label(win, bg="black")
label.pack()
label.bind("<ButtonPress-1>", start_drawing)
label.bind("<B1-Motion>", draw)
label.bind("<ButtonRelease-1>", stop_drawing)

Button(win, text="◀", bg="gray", command=count_down).place(x=200, y=toolbar_pos_y, width=40)
Button(win, text="▶", bg="gray", command=count_up).place(x=260, y=toolbar_pos_y, width=40)
Button(win, text="Erase", bg="gray", command=erase).place(x=320, y=toolbar_pos_y, width=40)
draw_button = Button(win, text="Draw", bg="gray", command=toggle_drawing)
draw_button.place(x=button_pos['x'] + padding_x, y=toolbar_pos_y, width=40)
Button(win, text="Save", bg="gray", command=save_drawing).place(x=button_pos['x'] + 2 * padding_x, y=toolbar_pos_y,
                                                                width=40)
Button(win, text="Exit", bg="gray", command=close_drawing).place(x=button_pos['x'] + 3 * padding_x, y=toolbar_pos_y,
                                                                 width=40)

Radiobutton(win, text="Thick = 1", variable=Radivar, value=1, command=sel).place(x=360, y=toolbar_pos_y - 50)
Radiobutton(win, text="Thick = 2", variable=Radivar, value=2, command=sel).place(x=440, y=toolbar_pos_y - 50)
Radiobutton(win, text="Thick = 3", variable=Radivar, value=3, command=sel).place(x=520, y=toolbar_pos_y - 50)

# Create a listbox with a scrollbar for image selection
scrollbar = Scrollbar(win)
scrollbar.pack(side=tk.LEFT, fill=tk.Y)

image_listbox = Listbox(win, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE, height=toolbar_pos_y - 100)
for img_name in ListSourceImage:
    image_listbox.insert(tk.END, img_name)
image_listbox.pack(side=tk.LEFT, fill=tk.Y)
image_listbox.bind("<<ListboxSelect>>", on_image_select)

scrollbar.config(command=image_listbox.yview)

win.bind('<Control-z>', lambda event: undo())
win.bind('<Control-y>', lambda event: redo())
win.bind('<Command-z>', lambda event: undo())  # For Mac
win.bind('<Command-y>', lambda event: redo())  # For Mac

# TODO: Add arrow shortcut
win.bind('<Left>', lambda event: count_down())
win.bind('<Right>', lambda event: count_up())


def change_to_project_path(main_script_path: str):
    """
    Change the current working directory to the directory where the script is located.
    """
    script_path = os.path.dirname(os.path.abspath(main_script_path))
    current_path = os.getcwd()

    if current_path != script_path:
        os.chdir(script_path)
        print(f"change current path to: {os.getcwd()}")


if __name__ == "__main__":
    print(f"current path: {os.getcwd()}")
    # change_to_project_path(os.getcwd())
    switch(0)  # Load the first image initially
    win.mainloop()
