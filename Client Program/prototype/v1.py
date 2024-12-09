import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import subprocess
import os

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def browse_dialog(dialog_type, filetypes=None):
    if dialog_type == "directory":
        return filedialog.askdirectory()
    elif dialog_type == "file":
        return filedialog.askopenfilename(filetypes=filetypes)
    return None

def create_checkbox_frame(master, row, column, padx=10, pady=10, sticky="nsew"):
    frame = ctk.CTkFrame(master)
    frame.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
    return frame

def create_button(master, text, row, column, command=None, padx=10, pady=10):
    button = ctk.CTkButton(master, text=text, command=command)
    button.grid(row=row, column=column, padx=padx, pady=pady)
    return button

def create_path_input(master, width, row, column, padx=10, pady=10):
    path_input = ctk.CTkEntry(master, width=width)
    path_input.grid(row=row, column=column, padx=padx, pady=pady)
    return path_input

def create_checkbox(master, text, command, row, column, padx=10, pady=10, state="normal", selected=False):
    checkbox = ctk.CTkCheckBox(master=master, text=text, command=command, state=state)
    checkbox.grid(row=row, column=column, padx=padx, pady=pady)
    if selected:
        checkbox.select()
    return checkbox

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("V1")
        self.geometry(f"{1100}x580")
        self.minsize(1100, 580)

        # Layout configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar (not sure why this disappears when navigating to other screens, might remove)
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0, fg_color="#505050")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1) # Amount of side buttons +1

        # Side buttons
        self.sidebar_buttons = [
            ctk.CTkButton(self.sidebar_frame, text="Convert to SVS", command=self.show_KFB_TO_SVS_SCREEN),
            ctk.CTkButton(self.sidebar_frame, text="SVS to JPG", command=self.show_SVS_TO_JPG_SCREEN),
            ctk.CTkButton(self.sidebar_frame, text="Color Correction", command=self.sidebar_button_event),
            ctk.CTkButton(self.sidebar_frame, text="Analyse Input", command=self.sidebar_button_event),
        ]
        for index, button in enumerate(self.sidebar_buttons, start=1):
            button.grid(row=index, column=0, padx=20, pady=10)

    def sidebar_button_event(self):
        print("side button clicked")

    def show_KFB_TO_SVS_SCREEN(self):
        self.conversion_screen = KFB_TO_SVS_SCREEN(self)
        self.conversion_screen.grid(row=0, column=1, sticky="nsew")

    def show_SVS_TO_JPG_SCREEN(self):
        self.conversion_screen = SVS_TO_JPG_SCREEN(self)
        self.conversion_screen.grid(row=0, column=1, sticky="nsew")
    
class KFB_TO_SVS_SCREEN(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)

        # Master frames for checkboxes
        self.checkbox_frame_kfb = create_checkbox_frame(self, row=0, column=2)
        self.checkbox_frame_output = create_checkbox_frame(self, row=1, column=2)

        # KFB input
        self.select_kfb = create_button(self, text="KFB Input", row=0, column=0, command=self.browse_kfb)
        self.kfb_path = create_path_input(self, width=300, row=0, column=1)
        self.kfb_isDir = create_checkbox(master=self.checkbox_frame_kfb, text="isDir", command=self.kfb_checkbox_toggle, row=0, column=0, selected=True)
        self.kfb_isFile = create_checkbox(master=self.checkbox_frame_kfb, text="isFile", command=self.kfb_checkbox_toggle, row=0, column=1)

        # TIF output
        self.select_output = create_button(self, text="TIF Output", row=1, column=0, command=self.browse_output_directory)
        self.output_path = create_path_input(self, width=300, row=1, column=1)
        self.output_isDir = create_checkbox(master=self.checkbox_frame_output, text="isDir", command=None, row=0, column=0, state="disabled", selected=True)

        # Start Conversion Button
        self.start_button = create_button(self, text="Start Conversion", row=2, column=0, pady=20, command=self.start_conversion)

        self.kfb_checkbox_toggle()

    def kfb_checkbox_toggle(self):
        self.kfb_isFile.configure(state="disabled" if self.kfb_isDir.get() else "normal")
        self.kfb_isDir.configure(state="disabled" if self.kfb_isFile.get() else "normal")

    def browse_kfb(self):
        if self.kfb_isDir.get():
            selected_path = browse_dialog("directory")
        elif self.kfb_isFile.get():
            selected_path = browse_dialog("file", filetypes=[("KFB Files", "*.kfb")])
        else:
            messagebox.showerror("Error", "Please select 'isDir' or 'isFile'")
            return

        if selected_path:
            self.kfb_path.delete(0, tk.END)
            self.kfb_path.insert(0, selected_path)

    def browse_output_directory(self):
        if self.output_isDir.get():
            selected_path = browse_dialog("directory")
        if selected_path:
            self.output_path.delete(0, tk.END)
            self.output_path.insert(0, selected_path)

    def start_conversion(self):
        kfb_dir = self.kfb_path.get().strip()
        output_dir = self.output_path.get().strip()

        if not kfb_dir or not output_dir:
            messagebox.showerror("Error", "Empty directory is not allowed!")
            return

        # Add actual conversion logic here
        messagebox.showinfo("Success", "Conversion started successfully!")

class SVS_TO_JPG_SCREEN(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)

        # Master frames for checkboxes
        self.checkbox_frame_kfb = create_checkbox_frame(self, row=0, column=2)
        self.checkbox_frame_output = create_checkbox_frame(self, row=1, column=2)

        # SVS input
        self.select_svs = create_button(self, text="SVS Input", row=0, column=0, command=self.browse_svs)
        self.svs_path = create_path_input(self, width=300, row=0, column=1)
        self.svs_isDir = create_checkbox(master=self.checkbox_frame_kfb, text="isDir", command=self.svs_checkbox_toggle, row=0, column=1, selected=True)
        self.svs_isFile = create_checkbox(master=self.checkbox_frame_kfb, text="isFile", command=self.svs_checkbox_toggle, row=0, column=2)

        # JPG output
        self.select_output = create_button(self, text="JPG Output", row=1, column=0, command=self.browse_output_directory)
        self.output_path = create_path_input(self, width=300, row=1, column=1)
        self.output_isDir = create_checkbox(master=self.checkbox_frame_output, text="isDir", command=None, row=1, column=2, state="disabled", selected=True)

        # Start Conversion Button
        self.start_button = create_button(self, text="Start Conversion", row=2, column=0, pady=20, command=self.start_conversion)

        self.svs_checkbox_toggle()

    def svs_checkbox_toggle(self):
        self.svs_isFile.configure(state="disabled" if self.svs_isDir.get() else "normal")
        self.svs_isDir.configure(state="disabled" if self.svs_isFile.get() else "normal")

    def browse_svs(self):
        if self.svs_isDir.get():
            selected_path = browse_dialog("directory")
        elif self.svs_isFile.get():
            selected_path = browse_dialog("file", filetypes=[("SVS Files", "*.svs")])
        else:
            messagebox.showerror("Error", "Please select 'isDir' or 'isFile'")
            return

        if selected_path:
            self.svs_path.delete(0, tk.END)
            self.svs_path.insert(0, selected_path)

    def browse_output_directory(self):
        if self.output_isDir.get():
            selected_path = browse_dialog("directory")
        if selected_path:
            self.output_path.delete(0, tk.END)
            self.output_path.insert(0, selected_path)

    def start_conversion(self):
        svs_dir = self.svs_path.get()
        output_dir = self.output_path.get()

        if not svs_dir or not output_dir:
            messagebox.showerror("Error", "Empty directory is not allowed!")
            return

        try:
            script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_svs_to_jpg')
            script_file = "convertSVS.py"

            if not os.path.isdir(script_dir):
                raise NotADirectoryError(f"Directory does not exist: {script_dir}")

            subprocess.run(
                [
                    "python",
                    script_file,
                    svs_dir,
                    output_dir,
                    "1024",  # Tile size
                    "0",     # Level
                    "4"      # Max workers
                ],
                cwd=script_dir,
                check=True
            )
            messagebox.showinfo("Success", "Conversion complete!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Conversion failed:\n{e}")
        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find the external script. Please check the path.")
        except NotADirectoryError as e:
            messagebox.showerror("Error", f"Invalid directory:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error:\n{e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
