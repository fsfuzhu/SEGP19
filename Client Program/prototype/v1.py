import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import os
import pyvips
from concurrent.futures import ThreadPoolExecutor

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

def file_input_criteria_toggle(isDir, isFile, dirCheckbox, fileCheckbox):
    fileCheckbox.configure(state="disabled" if isDir.get() else "normal")
    dirCheckbox.configure(state="disabled" if isFile.get() else "normal")

def browse_input(isDir, isFile, pathDialog, file_type):
    if isDir.get():
        selected_path = browse_dialog("directory")
    elif isFile.get():
        selected_path = browse_dialog("file", filetypes=[(f"{file_type} Files", f"*.{file_type.lower()}")])
    else:
        messagebox.showerror("Error", "Please select 'isDir' or 'isFile'")
        return
    
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

def browse_output_directory(isDir, pathDialog):
    if isDir.get():
        selected_path = browse_dialog("directory")
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

def convert_svs_to_jpg_tiles_parallel(input_path, output_dir, tile_size=1024, level=0, max_workers=4):
    """
    Split an SVS file into multiple JPG tiles in parallel.

    Parameters:
    - input_path: Input SVS file path
    - output_dir: Directory for output JPG files
    - tile_size: Size of each tile (default is 1024x1024 pixels)
    - level: Image level to read (default 0 is the highest resolution)
    - max_workers: Number of threads for parallel processing (default is 4)
    """

    svs_files = []
    if os.path.isdir(input_path):
        svs_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.svs')]
        if not svs_files:
            messagebox.showerror("Error", "No SVS files found in specific directory")
    else:
        svs_files = [input_path]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        for svs_file in svs_files:
            file_name = os.path.splitext(os.path.basename(svs_file))[0]
            file_output_dir = os.path.join(output_dir, file_name)
            os.makedirs(file_output_dir, exist_ok=True)

            # Load the SVS file
            image = pyvips.Image.new_from_file(svs_file, access='sequential')

            # Calculate scale factor
            scale_str = image.get('openslide.level[{}].downsample'.format(level))
            if scale_str is None:
                scale = 1.0 # Default to 1.0 if no downscale factor is specified
            else:
                scale = float(scale_str) # Convert string to float

            # Get dimensions for the specified level
            width = int(image.width / scale)
            height = int(image.height / scale)
            # print(f"Image dimensions at level {level}: {width}x{height}")

            # Calculate number of tiles
            tiles_x = (width + tile_size - 1) // tile_size
            tiles_y = (height + tile_size - 1) // tile_size
            # print(f"Dividing into {tiles_x} x {tiles_y} = {tiles_x * tiles_y} tiles")

            # Process tiles in parallel using a thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for ty in range(tiles_y):
                    for tx in range(tiles_x):
                        executor.submit(process_tile, image, scale, tx, ty, tile_size, level, file_output_dir)

        messagebox.showinfo("Success", "Conversion complete!")

    except Exception as e:
        messagebox.showerror("Error", f"Conversion failed: {e}")

def process_tile(image, scale, tx, ty, tile_size, level, output_dir):
    """
    Process a single tile and save it as a JPG file.
    """
    try:
        x = tx * tile_size * scale
        y = ty * tile_size * scale
        w = tile_size * scale
        h = tile_size * scale

        # Extract the region
        region = image.crop(x, y, w, h).resize(1 / scale)

        # Convert to RGB mode
        if region.bands == 4:
            region = region[:3]  # Remove the alpha channel
        elif region.bands == 1:
            region = region.colourspace("srgb")

        # Define output filename
        output_filename = f"tile_{ty}_{tx}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save as JPG file
        region.write_to_file(output_path, Q=100)  # Q=90 for JPEG quality
        print(f"Saved tile: {output_path}")
    except Exception as e:
        print(f"Failed to process tile ({tx}, {ty}): {e}")

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
            # ("Convert to SVS", lambda: self._switch_to_screen(KFB_TO_SVS_SCREEN)),
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
        messagebox.showinfo("Success", "Feature coming soon!")

    def _switch_to_screen(self, screen_class):
        if self.active_screen:
            self.active_screen.destroy()

        self.active_screen = screen_class(self)
        self.active_screen.grid(row=0, column=1, sticky="nsew")

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
        self.select_output = create_button(self, text="TIF Output", row=1, column=0, command=self.browse_output)
        self.output_path = create_path_input(self, width=300, row=1, column=1)
        self.output_isDir = create_checkbox(master=self.checkbox_frame_output, text="isDir", command=None, row=0, column=0, state="disabled", selected=True)

        # Start Conversion Button
        self.start_button = create_button(self, text="Start Conversion", row=2, column=0, pady=20, command=self.start_conversion)

        self.kfb_checkbox_toggle()

    def kfb_checkbox_toggle(self):
        file_input_criteria_toggle(self.kfb_isDir, self.kfb_isFile, self.kfb_isDir, self.kfb_isFile)

    def browse_kfb(self):
        browse_input(self.kfb_isDir, self.kfb_isFile, self.kfb_path, "KFB")

    def browse_output(self):
        browse_output_directory(self.output_isDir, self.output_path)

    def start_conversion(self):
        kfb_dir = self.kfb_path.get().strip()
        output_dir = self.output_path.get().strip()

        if not kfb_dir or not output_dir:
            messagebox.showerror("Error", "Empty directory is not allowed!")
            return

        # Add actual conversion logic here
        messagebox.showinfo("Success", "Feature coming soon!")

class SVS_TO_JPG_SCREEN(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)
        self.configure(fg_color="#2c2f33")  # Background color

        # Title
        self.title_label = ctk.CTkLabel(self, text="SVS to JPG Converter", font=("Arial", 20, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=3, pady=(20, 10), sticky="n")

        # SVS Input Section
        self.svs_label = ctk.CTkLabel(self, text="SVS Input", font=("Arial", 14))
        self.svs_label.grid(row=1, column=0, padx=10, pady=(10, 5), sticky="w")

        self.svs_path = create_path_input(self, width=300, row=1, column=1)
        self.select_svs = create_button(self, text="Browse", row=1, column=2, command=self.browse_svs)
        self.checkbox_frame_kfb = create_checkbox_frame(self, row=2, column=1)
        self.svs_isDir = create_checkbox(master=self.checkbox_frame_kfb, text="isDir", command=self.svs_checkbox_toggle, row=0, column=0, selected=True)
        self.svs_isFile = create_checkbox(master=self.checkbox_frame_kfb, text="isFile", command=self.svs_checkbox_toggle, row=0, column=1)

        # JPG Output Section
        self.output_label = ctk.CTkLabel(self, text="JPG Output", font=("Arial", 14))
        self.output_label.grid(row=3, column=0, padx=10, pady=(10, 5), sticky="w")

        self.output_path = create_path_input(self, width=300, row=3, column=1)
        self.select_output = create_button(self, text="Browse", row=3, column=2, command=self.browse_output)
        self.checkbox_frame_output = create_checkbox_frame(self, row=4, column=1)
        self.output_isDir = create_checkbox(master=self.checkbox_frame_output, text="isDir", command=None, row=0, column=0, state="disabled", selected=True)

        # Start Conversion Button
        self.start_button = create_button(self, text="Start Conversion", row=5, column=1, pady=20, command=self.start_conversion)
        self.start_button.configure(fg_color="#7289da", hover_color="#5b6eae")

        self.svs_checkbox_toggle()

    def svs_checkbox_toggle(self):
        file_input_criteria_toggle(self.svs_isDir, self.svs_isFile, self.svs_isDir, self.svs_isFile)

    def browse_svs(self):
        browse_input(self.svs_isDir, self.svs_isFile, self.svs_path, "SVS")

    def browse_output(self):
        browse_output_directory(self.output_isDir, self.output_path)

    def start_conversion(self):
        svs_dir = self.svs_path.get()
        output_dir = self.output_path.get()

        if not svs_dir or not output_dir:
            messagebox.showerror("Error", "Empty directory is not allowed!")
            return

        convert_svs_to_jpg_tiles_parallel(svs_dir, output_dir, 1024, 0, 4)

if __name__ == "__main__":
    app = App()
    app.mainloop()
