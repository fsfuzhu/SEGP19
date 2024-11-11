import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

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
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Side buttons
        self.sidebar_buttons = [
            ctk.CTkButton(self.sidebar_frame, text="Convert to SVS", command=self.show_conversion_screen),
            ctk.CTkButton(self.sidebar_frame, command=self.sidebar_button_event),
            ctk.CTkButton(self.sidebar_frame, command=self.sidebar_button_event),
        ]
        for index, button in enumerate(self.sidebar_buttons, start=1):
            button.grid(row=index, column=0, padx=20, pady=10)

    def sidebar_button_event(self):
        print("side button clicked")

    def show_conversion_screen(self):
        self.conversion_screen = ConversionScreen(self)
        self.conversion_screen.grid(row=0, column=1, sticky="nsew")
    
class ConversionScreen(ctk.CTkFrame):
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

if __name__ == "__main__":
    app = App()
    app.mainloop()
