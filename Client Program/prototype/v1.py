import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import subprocess
import os

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("V1")
        self.geometry("1100x580")
        self.minsize(1100, 580)

        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # UI
        self._create_sidebar()
        self.active_screen = None

    def _create_sidebar(self):
        sidebar_buttons_data = [
            ("Convert to SVS", lambda: self._switch_to_screen(KFB_TO_SVS_SCREEN)),
            ("SVS to JPG", lambda: self._switch_to_screen(SVS_TO_JPG_SCREEN)),
            ("Color Correction", self.sidebar_button_event),
            ("Analyze Input", self.sidebar_button_event),
        ]

        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0, fg_color="#505050")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(len(sidebar_buttons_data) + 1, weight=1)

        for index, (text, command) in enumerate(sidebar_buttons_data, start=1):
            button = ctk.CTkButton(self.sidebar_frame, text=text, command=command)
            button.grid(row=index, column=0, padx=20, pady=10)

    def sidebar_button_event(self):
        print("side button clicked")

    def _switch_to_screen(self, screen_class):
        if self.active_screen:
            self.active_screen.destroy()

        self.active_screen = screen_class(self)
        self.active_screen.grid(row=0, column=1, sticky="nsew")

class KFB_TO_SVS_SCREEN(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)

        #master frame for checkboxes
        self.checkbox_frame_kfb = ctk.CTkFrame(self)
        self.checkbox_frame_kfb.grid(row=0, column=2, padx=20, pady=10, sticky="nsew")
        self.checkbox_frame_output = ctk.CTkFrame(self)
        self.checkbox_frame_output.grid(row=1, column=2, padx=20, pady=10, sticky="nsew")

        self.select_kfb = ctk.CTkButton(self, text="KFB Input", command=self.browse_kfb)
        self.select_kfb.grid(row=0, column=0, padx=20, pady=10)
        self.kfb_path = ctk.CTkEntry(self, width=300)
        self.kfb_path.grid(row=0, column=1, padx=20, pady=10)
        self.kfb_isDir = ctk.CTkCheckBox(master=self.checkbox_frame_kfb, text="isDir", command=self.kfb_checkbox_toggle)
        self.kfb_isDir.grid(row=0, column=2, padx=20, pady=10)
        self.kfb_isFile = ctk.CTkCheckBox(master=self.checkbox_frame_kfb, text="isFile", command=self.kfb_checkbox_toggle)
        self.kfb_isFile.grid(row=0, column=3, padx=20, pady=10)

        self.select_output = ctk.CTkButton(self, text="TIF Output", command=self.browse_output_directory)
        self.select_output.grid(row=1, column=0, padx=20, pady=10)
        self.output_path = ctk.CTkEntry(self, width=300)
        self.output_path.grid(row=1, column=1, padx=20, pady=10)
        self.output_isDir = ctk.CTkCheckBox(master=self.checkbox_frame_output, text="isDir")
        self.output_isDir.grid(row=1, column=2, padx=20, pady=10)
        self.output_isDir.select()
        self.output_isDir.configure(state="disabled")

        self.start_button = ctk.CTkButton(self, text="Start Conversion", command=self.start_conversion)
        self.start_button.grid(row=2, column=0, padx=20, pady=10)

    def kfb_checkbox_toggle(self):
        if self.kfb_isDir.get():
            self.kfb_isFile.configure(state="disabled")
        else:
            self.kfb_isFile.configure(state="normal")
            
        if self.kfb_isFile.get():
            self.kfb_isDir.configure(state="disabled")
        else:
            self.kfb_isDir.configure(state="normal")

    #add error handling for files not found
    def browse_kfb(self):
        selected_path = None
        if self.kfb_isDir.get():
            selected_path = filedialog.askdirectory()
        elif self.kfb_isFile.get():
            selected_path = filedialog.askopenfilename(filetypes=[("KFB Files", "*.kfb")])
        else:
            print("Please select checkbox") #this needs to be printed at user interface later

        if selected_path:
            self.kfb_path.delete(0, tk.END)
            self.kfb_path.insert(0, selected_path)

    def browse_output_directory(self):
        selected_path = None
        if self.output_isDir.get():
            selected_path = filedialog.askdirectory()
            self.output_path.delete(0, tk.END)
            self.output_path.insert(0, selected_path)

    def start_conversion(self):
        #temp
        kfb_dir = self.kfb_path.get()
        output_dir = self.output_path.get()

        if not kfb_dir or not output_dir:
            tk.messagebox.showerror("Error", "Empty directory is not allowed!")
            return

class SVS_TO_JPG_SCREEN(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)

        #master frame for checkboxes
        self.checkbox_frame_kfb = ctk.CTkFrame(self)
        self.checkbox_frame_kfb.grid(row=0, column=2, padx=20, pady=10, sticky="nsew")
        self.checkbox_frame_output = ctk.CTkFrame(self)
        self.checkbox_frame_output.grid(row=1, column=2, padx=20, pady=10, sticky="nsew")

        self.select_svs = ctk.CTkButton(self, text="SVS Input", command=self.browse_svs)
        self.select_svs.grid(row=0, column=0, padx=20, pady=10)
        self.svs_path = ctk.CTkEntry(self, width=300)
        self.svs_path.grid(row=0, column=1, padx=20, pady=10)
        # self.svs_isDir = ctk.CTkCheckBox(master=self.checkbox_frame_kfb, text="isDir", command=self.kfb_checkbox_toggle) //dont remove
        self.svs_isDir = ctk.CTkCheckBox(master=self.checkbox_frame_kfb, text="isDir", state="disabled")
        self.svs_isDir.grid(row=0, column=2, padx=20, pady=10)
        # self.svs_isFile = ctk.CTkCheckBox(master=self.checkbox_frame_kfb, text="isFile", command=self.kfb_checkbox_toggle) //dont remove
        self.svs_isFile = ctk.CTkCheckBox(master=self.checkbox_frame_kfb, text="isFile")
        self.svs_isFile.select()
        self.svs_isFile.grid(row=0, column=3, padx=20, pady=10)

        self.select_output = ctk.CTkButton(self, text="JPG Output", command=self.browse_output_directory)
        self.select_output.grid(row=1, column=0, padx=20, pady=10)
        self.output_path = ctk.CTkEntry(self, width=300)
        self.output_path.grid(row=1, column=1, padx=20, pady=10)
        self.output_isDir = ctk.CTkCheckBox(master=self.checkbox_frame_output, text="isDir")
        self.output_isDir.grid(row=1, column=2, padx=20, pady=10)
        self.output_isDir.select()
        self.output_isDir.configure(state="disabled")

        self.start_button = ctk.CTkButton(self, text="Start Conversion", command=self.start_conversion)
        self.start_button.grid(row=2, column=0, padx=20, pady=10)

    def kfb_checkbox_toggle(self):
        if self.kfb_isDir.get():
            self.kfb_isFile.configure(state="disabled")
        else:
            self.kfb_isFile.configure(state="normal")
            
        if self.kfb_isFile.get():
            self.kfb_isDir.configure(state="disabled")
        else:
            self.kfb_isDir.configure(state="normal")

    def browse_svs(self):
        selected_path = None
        if self.svs_isDir.get():
            selected_path = filedialog.askdirectory()
        elif self.svs_isFile.get():
            selected_path = filedialog.askopenfilename(filetypes=[("SVS Files", "*.svs")])
        else:
            print("Please select checkbox") #this needs to be printed at user interface later

        if selected_path:
            self.svs_path.delete(0, tk.END)
            self.svs_path.insert(0, selected_path)

    def browse_output_directory(self):
        selected_path = None
        if self.output_isDir.get():
            selected_path = filedialog.askdirectory()
            self.output_path.delete(0, tk.END)
            self.output_path.insert(0, selected_path)

    def start_conversion(self):
        svs_dir = self.svs_path.get()
        output_dir = self.output_path.get()

        if not svs_dir or not output_dir:
            tk.messagebox.showerror("Error", "Empty directory is not allowed!")
            return

        try:
            # Resolve the absolute path to the 'split_svs_to_jpg' folder, relative to v1.py
            script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_svs_to_jpg')
            script_file = "convertSVS.py"

            # Validate the script directory exists
            if not os.path.isdir(script_dir):
                raise NotADirectoryError(f"Directory does not exist: {script_dir}")

            # Call the external script
            subprocess.run(
                [
                    "python",
                    script_file,  # Path to your external script
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
