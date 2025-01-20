import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import os
import pyvips
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from torchvision.models import resnet18
from torchvision import transforms

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

YOLO_CONFIDENCE_THRESHOLD = 0.5  # Adjust based on model performance
RESNET_CONFIDENCE_THRESHOLD = 0.0  # Adjust based on desired specificity
RESNET_MODEL_PATH = './models/resnet18_model.pth'
YOLO_MODEL_PATH = './models/yolov8x_model.pt'

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
            ("Analyze Input", lambda: self._switch_to_screen(analyze_JPG_SCREEN)),
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

        self.thread = threading.Thread(
            target=self.convert_svs_to_jpg_tiles_parallel,
            args=(svs_dir, output_dir, 1024, 0, 4),
            daemon=True
        )
        self.thread.start()

    def convert_svs_to_jpg_tiles_parallel(self, input_path, output_dir, tile_size=1024, level=0, max_workers=4):
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
                for ty in range(tiles_y):
                    for tx in range(tiles_x):
                        self.process_tile(image, scale, tx, ty, tile_size, level, file_output_dir)

            messagebox.showinfo("Success", "Conversion complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {e}")

    def process_tile(self, image, scale, tx, ty, tile_size, level, output_dir):
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

class analyze_JPG_SCREEN(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)
        self.configure(fg_color="#2c2f33")  # Background color

        # Title
        self.title_label = ctk.CTkLabel(self, text="Analyze Images", font=("Arial", 20, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=3, pady=(20, 10), sticky="n")

        # JPG Input Section
        self.jpg_label = ctk.CTkLabel(self, text="JPG Folder", font=("Arial", 14))
        self.jpg_label.grid(row=1, column=0, padx=10, pady=(10, 5), sticky="w")

        self.jpg_path = create_path_input(self, width=300, row=1, column=1)
        self.select_jpg = create_button(self, text="Browse", row=1, column=2, command=self.browse_jpg)
        self.checkbox_frame_jpg = create_checkbox_frame(self, row=2, column=1)
        self.isCPU = create_checkbox(master=self.checkbox_frame_jpg, text="CPU", command=self.device_checkbox_toggle, row=0, column=0, selected=True)
        self.isGPU = create_checkbox(master=self.checkbox_frame_jpg, text="GPU", command=self.device_checkbox_toggle, row=0, column=1)

        # Output Section
        self.output_label = ctk.CTkLabel(self, text="AI Output Destination", font=("Arial", 14))
        self.output_label.grid(row=3, column=0, padx=10, pady=(10, 5), sticky="w")

        self.output_path = create_path_input(self, width=300, row=3, column=1)
        self.select_output = create_button(self, text="Browse", row=3, column=2, command=self.browse_output)
        self.checkbox_frame_output = create_checkbox_frame(self, row=4, column=1)
        self.output_isDir = create_checkbox(master=self.checkbox_frame_output, text="isDir", command=None, row=0, column=0, state="disabled", selected=True)

        # Hidden Variable
        self.jpg_isDir = create_checkbox(master=None, text="isDir", command=None, row=0, column=0, selected=True)
        self.jpg_isDir.grid_forget()

        # Start Conversion Button
        self.start_button = create_button(self, text="Start Conversion", row=5, column=1, pady=20, command=self.threading_analyze)
        self.start_button.configure(fg_color="#7289da", hover_color="#5b6eae")

        # Additional Text
        self.notes = ctk.CTkLabel(self, text="If GPU can't be enabled, it means CUDA isn't detected in system.", font=("Arial", 15))
        self.notes.grid(row=6, column=0, columnspan=3, pady=(20, 10), sticky="n")

        # Select CPU as default first
        self.device = torch.device("cpu")
        self.device_checkbox_toggle()

    def browse_jpg(self):
        browse_input(self.jpg_isDir, None, self.jpg_path, None)

    def browse_output(self):
        browse_output_directory(self.output_isDir, self.output_path)

    def device_checkbox_toggle(self):
        self.isGPU.configure(state="disabled" if not torch.cuda.is_available() or self.isCPU.get() else "normal")
        self.isCPU.configure(state="disabled" if self.isGPU.get() else "normal")

    def threading_analyze(self):
        self.thread = threading.Thread(
            target=self.start_analyzing,
            args=(),
            daemon=True
        )
        self.thread.start()

    def load_yolo_model(self):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        return yolo_model
    
    def load_resnet_model(self, num_classes):
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model

    # Function to perform cell detection using YOLO
    def detect_cells(self, yolo_model, image):
        results = yolo_model.predict(source=image, save=False)
        boxes = []
        for result in results:
            if result.boxes is None:
                continue  # No boxes detected in this result

            # Iterate over each bounding box
            for i, box in enumerate(result.boxes.xyxy):
                # Move the box tensor to CPU and convert to NumPy
                box_np = box.cpu().numpy()
                
                # Check for NaN values
                if np.isnan(box_np).any():
                    continue  # Skip invalid boxes

                # Convert box coordinates to integers
                x1, y1, x2, y2 = map(int, box_np)
                conf = float(result.boxes.conf[i].cpu().numpy()) if len(result.boxes.conf) > 0 else 0.0
                if conf > YOLO_CONFIDENCE_THRESHOLD:
                    boxes.append({
                        'bbox': [x1, y1, x2, y2]
                    })
        return boxes

    # Preprocessing for ResNet model
    def preprocess_cell_image(self, cell_image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        ])
        return preprocess(cell_image)

    # Function to classify cropped cells using ResNet
    def classify_cells(self, resnet_model, cell_images):
        cell_tensors = []
        for cell_img in cell_images:
            processed = self.preprocess_cell_image(cell_img)
            cell_tensors.append(processed)
        cell_batch = torch.stack(cell_tensors).to(self.device)
        with torch.no_grad():
            outputs = resnet_model(cell_batch)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
        return predictions.cpu().numpy(), confidences.cpu().numpy()

    def start_analyzing(self):
        jpg_dir = self.jpg_path.get()
        output_dir = self.output_path.get()

        if not jpg_dir or not output_dir:
            messagebox.showerror("Error", "Empty directory is not allowed!")
            return

        if self.isCPU.get():
            self.device = torch.device("cpu")
        elif self.isGPU.get():
            self.device = torch.device("cuda")
        else:
            messagebox.showerror("Error", "Please select either CPU or GPU to process images")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        CLASS_NAMES = ['abnormal', 'benign', 'normal']
        CLASS_COLOURS = {
            "normal": (100, 131, 54),    # Green for Normal
            "abnormal": (28, 32, 190),    # Red for Abnormal
            "benign":  (204, 102, 0),      # Blue for Benigh
        }

        yolo_model = self.load_yolo_model()
        resnet_model = self.load_resnet_model(len(CLASS_NAMES))
        normal_total, abnormal_total, benign_total = 0, 0, 0

        for img_filename in os.listdir(jpg_dir):
            img_path = os.path.join(jpg_dir, img_filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image {img_filename}")
                continue

            # Detect cells
            detections = self.detect_cells(yolo_model, image)

            # List to hold cropped cell images
            cropped_cells = []
            boxes_to_draw = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cell_image = image[y1:y2, x1:x2]
                cropped_cells.append(cell_image)
                boxes_to_draw.append({
                    'bbox': det['bbox'],
                    'class': None,  # Placeholder, will be filled after classification
                    'class_confidence': None,  # Placeholder
                })

            if len(cropped_cells) == 0:
                continue

            # Classify cells
            predictions, confidences = self.classify_cells(resnet_model, cropped_cells)

            # Filter and annotate detections
            for idx, box in enumerate(boxes_to_draw):
                class_idx = predictions[idx]
                class_confidence = confidences[idx]
                if class_confidence < RESNET_CONFIDENCE_THRESHOLD:
                    continue  # Skip low-confidence predictions
                
                if class_idx == 0:
                    abnormal_total += 1
                elif class_idx == 1:
                    benign_total += 1
                else:
                    normal_total += 1
                class_name = CLASS_NAMES[class_idx]
                box['class'] = class_name
                box['class_confidence'] = class_confidence

                # Draw bounding box and label on the image
                x1, y1, x2, y2 = map(int, box['bbox'])
                label = f"{class_name}: {class_confidence:.2f}"
                color = CLASS_COLOURS.get(class_name, (255, 255, 255))  # Default to white if class ID not in class_colors
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1 + 1, y1 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            output_path = os.path.join(output_dir, img_filename)
            cv2.imwrite(output_path, image)
            print(f"Processed and saved annotated image: {output_path}")

        total_cells = normal_total + abnormal_total + benign_total
        print(f"Normal: {normal_total} ({(normal_total/total_cells)*100}%), Abnormal: {abnormal_total} ({(abnormal_total/total_cells)*100}%), Benign: {benign_total} ({(benign_total/total_cells)*100}%)")

if __name__ == "__main__":
    app = App()
    app.mainloop()
