import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import json
from tkinter import filedialog

# Global variables
image_list = []
current_index = 0
current_directory = ""
zoom_scale = 0.5
current_image = None
image_id = None  # ID for the image on the canvas 
image_pos_x, image_pos_y = 400, 250
start_x, start_y = None, None
added_notes = []

def load_images(directory):
    global image_list, current_index, current_directory, zoom_scale, image_pos_x, image_pos_y
    current_directory = directory
    image_list = []
    current_index = 0
    zoom_scale = 0.5
    image_pos_x, image_pos_y = 400, 250

    canvas.delete("all")

    try:
        image_list = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_list.sort()
    except FileNotFoundError:
        show_message(f"Directory '{directory}' not found.")
        return

    if not image_list:
        show_message("No images found.")
        return

    display_image(current_index)

def display_image(index):
    global zoom_scale, current_image, image_id, image_pos_x, image_pos_y
    if image_id is not None:
        canvas.delete(image_id)

    image_file = image_list[index]
    file_label.config(text=image_file)

    #update tag (finally works)
    image_name_label.config(text=image_file)

    img_path = os.path.join(current_directory, image_file)
    current_image = Image.open(img_path)
    resized_image = current_image.resize((int(current_image.width * zoom_scale), int(current_image.height * zoom_scale)), Image.LANCZOS)
    photo = ImageTk.PhotoImage(resized_image)

    image_id = canvas.create_image(image_pos_x, image_pos_y, image=photo, anchor=tk.CENTER)
    canvas.image = photo

    update_buttons()

def zoom(event):
    global zoom_scale
    if event.delta > 0:
        zoom_scale *= 1.1
    elif event.delta < 0 and zoom_scale > 0.1:
        zoom_scale /= 1.1

    display_image(current_index)

def start_drag(event):
    global start_x, start_y
    start_x = event.x
    start_y = event.y

def drag_image(event):
    global image_pos_x, image_pos_y, start_x, start_y

    if start_x is not None and start_y is not None:
        dx = event.x - start_x
        dy = event.y - start_y
        canvas.move(image_id, dx, dy)
        image_pos_x += dx
        image_pos_y += dy
        start_x, start_y = event.x, event.y

def show_message(message):
    canvas.delete("all")
    canvas.create_text(400, 300, text=message, font=("Arial", 12), fill="black")

def next_image():
    global current_index, zoom_scale, image_pos_x, image_pos_y
    if current_index < len(image_list) - 1:
        current_index += 1
        zoom_scale = 0.5
        image_pos_x, image_pos_y = 400, 250
        display_image(current_index)

def prev_image():
    global current_index, zoom_scale, image_pos_x, image_pos_y
    if current_index > 0:
        current_index -= 1
        zoom_scale = 0.5
        image_pos_x, image_pos_y = 400, 250
        display_image(current_index)

def update_buttons():
    prev_button["state"] = tk.NORMAL if current_index > 0 else tk.DISABLED
    next_button["state"] = tk.NORMAL if current_index < len(image_list) - 1 else tk.DISABLED

def add_note():
    image_name = file_label.cget("text")
    cell_id = cell_id_entry.get()
    classification = classification_var.get()
    
    folder_name = "AP18" if "AP18" in current_directory else "AP19"

    note = {
        "image_name": image_name, 
        "cell_id": cell_id, 
        "classification": classification, 
        "folder": folder_name
    }
    added_notes.append(note)
    update_notes_list()
    cell_id_entry.delete(0, tk.END)
    classification_var.set("Normal")

def delete_note():
    selected_index = notes_listbox.curselection()

    if not selected_index:
        show_message("Please select a note to delete.")
        return

    #double confirm
    confirm = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete the selected note?")
    
    if confirm:
        selected_index = selected_index[0]
        del added_notes[selected_index]
        update_notes_list()

def update_notes_list():
    notes_listbox.delete(0, tk.END)

    for note in added_notes:
        note_text = (f"Image: {note['image_name']} | Cell ID: {note['cell_id']} | "
                     f"Classification: {note['classification']} | Source: {note['folder']}")
        notes_listbox.insert(tk.END, note_text)

def export_notes_to_json():
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    
    if not file_path:
        return

    try:
        with open(file_path, 'w') as json_file:
            json.dump(added_notes, json_file, indent=4)
        show_message(f"Notes exported successfully to {file_path}")
    except Exception as e:
        show_message(f"Error exporting notes: {e}")

#window
window = tk.Tk()
window.title("Pathologists Quick Review [UI]")
window_width = 1200
window_height = 900

window.geometry(f"{window_width}x{window_height}")
window.update_idletasks()

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

#mid screen
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

image_frame = tk.Frame(window)
image_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(image_frame, width=800, height=500, bg="white")
canvas.pack()

file_label = tk.Label(image_frame, text="", font=("Arial", 10))
file_label.pack()

canvas.bind("<MouseWheel>", zoom)
canvas.bind("<ButtonPress-1>", start_drag)
canvas.bind("<B1-Motion>", drag_image)

button_frame = tk.Frame(window)
button_frame.pack(side=tk.BOTTOM, pady=10)

button_ap18 = tk.Button(button_frame, text="AP18", command=lambda: load_images("sources/AP18"))
button_ap18.pack(side=tk.LEFT, padx=10)

button_ap19 = tk.Button(button_frame, text="AP19", command=lambda: load_images("sources/AP19"))
button_ap19.pack(side=tk.LEFT, padx=10)

nav_frame = tk.Frame(window)
nav_frame.pack(side=tk.BOTTOM, pady=10)

prev_button = tk.Button(nav_frame, text="Previous", command=prev_image, state=tk.DISABLED)
prev_button.pack(side=tk.LEFT, padx=10)

next_button = tk.Button(nav_frame, text="Next", command=next_image, state=tk.DISABLED)
next_button.pack(side=tk.LEFT, padx=10)

note_frame = tk.Frame(window)
note_frame.pack(side=tk.LEFT, padx=20, pady=20)

tk.Label(note_frame, text="Cell ID (number only):").pack()
cell_id_entry = tk.Entry(note_frame)
cell_id_entry.pack()

tk.Label(note_frame, text="Image Name:").pack()
image_name_label = tk.Label(note_frame, text="")
image_name_label.pack()

classification_var = tk.StringVar(value="Normal")
tk.Label(note_frame, text="Classification:").pack()
tk.Radiobutton(note_frame, text="Normal", variable=classification_var, value="Normal").pack(anchor=tk.W)
tk.Radiobutton(note_frame, text="Abnormal", variable=classification_var, value="Abnormal").pack(anchor=tk.W)
tk.Radiobutton(note_frame, text="Benign", variable=classification_var, value="Benign").pack(anchor=tk.W)
tk.Radiobutton(note_frame, text="Rubbish", variable=classification_var, value="Rubbish").pack(anchor=tk.W)

add_note_button = tk.Button(note_frame, text="Add Note", command=add_note)
add_note_button.pack(pady=10)

notes_frame = tk.Frame(window)
notes_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.Y)

tk.Label(notes_frame, text="Added Notes", font=("Arial", 12)).pack()

scrollbar = tk.Scrollbar(notes_frame)

notes_listbox = tk.Listbox(notes_frame, width=100, height=25)
notes_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
notes_listbox.bind("<Delete>", lambda event: delete_note())

scrollbar.config(command=notes_listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

button_frame = tk.Frame(notes_frame)
button_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

delete_note_button = tk.Button(button_frame, text="Delete Selected Note", command=delete_note)
delete_note_button.pack(side=tk.LEFT, padx=10)

export_button = tk.Button(button_frame, text="Export Notes to JSON", command=export_notes_to_json)
export_button.pack(side=tk.LEFT, padx=10)

window.mainloop()
