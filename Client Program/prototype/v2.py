import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def create_ctk_frame(master, row, column, columnspan, padx=10, pady=10, sticky="nsew"):
    frame = ctk.CTkFrame(master)
    frame.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky)
    return frame

def create_radiobutton(master, text, row, column, variable, value, padx=10, pady=10):
    radiobutton = ctk.CTkRadioButton(master=master, text=text, variable=variable, value=value)
    radiobutton.grid(row=row, column=column, padx=padx, pady=pady)
    return radiobutton

def create_path_input(master, width, row, column, padx=10, pady=10):
    path_input = ctk.CTkEntry(master, width=width)
    path_input.grid(row=row, column=column, padx=padx, pady=pady)
    return path_input

def create_button(master, text, row, column, command=None, padx=10, pady=10):
    button = ctk.CTkButton(master, text=text, command=command)
    button.grid(row=row, column=column, padx=padx, pady=pady)
    return button

def browse_dialog(dialog_type, filetypes=None):
    if dialog_type == "directory":
        return filedialog.askdirectory()
    elif dialog_type == "file":
        return filedialog.askopenfilename(filetypes=filetypes)
    return None

def browse_input(pathDialog, file_type):
    selected_path = browse_dialog("file", filetypes=[(f"{file_type} Files", f"*.{file_type.lower()}")])
    
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

def browse_output_directory(pathDialog):
    selected_path = browse_dialog("directory")
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("V2")
        self.geometry("550x532")
        self.resizable(False, False)

        # Grid config
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # SVS Input Section (row 0)
        self.select_svs = create_button(self, text="Browse SVS", row=0, column=0, padx=40, pady=20, command=self.browse_svs)
        self.svs_path = create_path_input(self, width=250, row=0, column=1)

        # Browse JPG Output (row 1)
        self.output_path = create_button(self, text="JPG Output", row=1, column=0, padx=40, pady=20, command=self.browse_output)
        self.jpg_path = create_path_input(self, width=250, row=1, column=1)

        # Select Basic/Advanced Detection (row 2)
        self.radio_var = tk.StringVar(value="Basic")
        self.radiobutton_frame = create_ctk_frame(self, row=2, column=0, columnspan=2, padx=(40, 10), sticky="ew")
        self.label_radio_group = ctk.CTkLabel(master=self.radiobutton_frame, text="Model Detection: ", font=("Calibri", 15))
        self.label_radio_group.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.basic_detect = create_radiobutton(master=self.radiobutton_frame, text="Basic", row=0, column=1, variable=self.radio_var, value="Basic")
        self.advance_detect = create_radiobutton(master=self.radiobutton_frame, text="Advanced", row=0, column=2,variable=self.radio_var, value="Advanced")

        # Start/Cancel Button (row 3)
        self.start_stop_button = ctk.CTkButton(self, text="Start", command=None, font=("Calibri", 50))
        self.start_stop_button.grid(row=3, column=0, columnspan=2, padx=(40, 10), pady=20, sticky="ew")
        
        # Progress Bar (row 4)
        self.progress_frame = create_ctk_frame(self, row=4, column=0, columnspan=2, padx=(40, 10), sticky="ew")
        self.progress_label = ctk.CTkLabel(master=self.progress_frame, text="Progress:", font=("Calibri", 20))
        self.progress_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.progress_bar = ctk.CTkProgressBar(master=self.progress_frame)
        self.progress_bar.grid(row=0, column=1, padx=(10, 10), pady=10, sticky="ew")
        self.progress_frame.grid_columnconfigure(1, weight=1)
        self.progress_bar.set(0.3)


    def browse_svs(self):
        browse_input(self.svs_path, "SVS")

    def browse_output(self):
        browse_output_directory(self.output_path)

if __name__ == "__main__":
    app = App()
    app.mainloop()