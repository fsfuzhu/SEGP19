import customtkinter as ctk
import torch.multiprocessing as mp
from tkinter import filedialog, messagebox
import threading
import torch
import os
import cv2
import tkinter as tk
import pyvips
from ultralytics import YOLO
from torchvision.models import resnet18
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import time
from multiprocessing import Process, Queue, Event
import multiprocessing as mp
import numpy as np
mp.set_start_method('spawn', force=True)

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

YOLO_CONFIDENCE_THRESHOLD = 0.5  # Adjust based on model performance
RESNET_CONFIDENCE_THRESHOLD = 0.0  # Adjust based on desired specificity
RESNET_MODEL_PATH = './models/resnet18_model.pth'
YOLO_MODEL_PATH = './models/yolov8x_model.pt'
CLASS_NAMES = ['abnormal', 'benign', 'normal']

def process_analyze(jpg_dir, output_dir, device, queue, stop_event):
    print("Current thread (process_analyze):", threading.current_thread())
    """
    This function runs in a separate process.
    It loads the models (on the specified device), processes images, and sends status messages to the queue.
    """
    current_dev = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(current_dev)
    print(f"GPU worker (process_analyze): CUDA context created on device {current_dev} - {dev_name}")

    # Load models in this process
    try:
        yolo_model = load_yolo_model(device=device)
    except Exception as e:
        queue.put(f"Error loading YOLO model: {e}")
        return

    try:
        resnet_model = load_resnet_model(num_classes=len(CLASS_NAMES), device=device)
    except Exception as e:
        print(f"Error loading ResNet model: {e}")
        queue.put(f"Error loading ResNet model: {e}")
        return

    try:
        current_dev = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(current_dev)
        print(f"CUDA context created: device {current_dev} - {dev_name}")
        # Allocate a dummy tensor to verify memory allocations
        dummy = torch.zeros(1, device='cuda')
        print("Dummy tensor allocated successfully.")
        print(torch.cuda.memory_summary(device=current_dev))
    except Exception as e:
        print("Error initializing CUDA context:", e)
    
    # Begin processing images
    total_start = time.time()
    normal_total, abnormal_total, benign_total = 0, 0, 0

    # Loop over images in the jpg_dir
    for img_filename in os.listdir(jpg_dir):
        if stop_event.is_set():
            queue.put("Processing cancelled by user.")
            return

        img_path = os.path.join(jpg_dir, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            queue.put(f"Failed to load image {img_filename}")
            continue

        # Run YOLO detection
        try:
            results = yolo_model.predict(source=image, save=False)
        except Exception as e:
            queue.put(f"Error during YOLO prediction on {img_filename}: {e}")
            continue

        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for i, box in enumerate(result.boxes.xyxy):
                box_np = box.cpu().numpy()
                if np.isnan(box_np).any():
                    continue
                x1, y1, x2, y2 = map(int, box_np)
                conf = float(result.boxes.conf[i].cpu().numpy()) if len(result.boxes.conf) > 0 else 0.0
                if conf > YOLO_CONFIDENCE_THRESHOLD:
                    boxes.append({'bbox': [x1, y1, x2, y2]})

        cropped_cells = []
        boxes_to_draw = []
        for det in boxes:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cell_image = image[y1:y2, x1:x2]
            cropped_cells.append(cell_image)
            boxes_to_draw.append({'bbox': det['bbox'], 'class': None, 'class_confidence': None})

        if len(cropped_cells) == 0:
            continue

        # Classify cells with ResNet
        try:
            predictions, confidences = classify_cells(cropped_cells, resnet_model, device)
            print("predictions:", predictions)
            print("confidences:", confidences)
        except Exception as e:
            print(f"Error during ResNet classification on {img_filename}: {e}")
            queue.put(f"Error during ResNet classification on {img_filename}: {e}")
            continue

        for idx, box in enumerate(boxes_to_draw):
            class_idx = predictions[idx]
            class_confidence = confidences[idx]
            if class_confidence < RESNET_CONFIDENCE_THRESHOLD:
                continue
            if class_idx == 0:
                abnormal_total += 1
            elif class_idx == 1:
                benign_total += 1
            else:
                normal_total += 1
            class_name = CLASS_NAMES[class_idx]
            box['class'] = class_name
            box['class_confidence'] = class_confidence

            # Draw the bounding box on the image
            x1, y1, x2, y2 = map(int, box['bbox'])
            label = f"{class_name}: {class_confidence:.2f}"
            # Choose a color based on class (you can define your own colors)
            color = (0, 255, 0) if class_name == "normal" else ((0, 0, 255) if class_name == "abnormal" else (255, 0, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1 + 1, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save the output image
        output_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(output_path, image)
        queue.put(f"Processed and saved {img_filename}")

    total_cells = normal_total + abnormal_total + benign_total
    total_time = time.time() - total_start
    queue.put(f"Total time: {total_time:.2f} seconds")
    if total_cells > 0:
        queue.put(f"Normal: {normal_total} ({(normal_total/total_cells)*100:.1f}%), "
                  f"Abnormal: {abnormal_total} ({(abnormal_total/total_cells)*100:.1f}%), "
                  f"Benign: {benign_total} ({(benign_total/total_cells)*100:.1f}%)")
    else:
        queue.put("No cells processed.")
    queue.put("Processing completed.")

# Simple function to perform cell classification
def classify_cells(cell_images, resnet_model, device, batch_size=4):
    predictions_list = []
    confidences_list = []
    with torch.no_grad():
        # Process in batches of size 'batch_size'
        for i in range(0, len(cell_images), batch_size):
            batch_imgs = cell_images[i:i+batch_size]
            # Preprocess each cell image
            cell_tensors = []
            for img in batch_imgs:
                processed = preprocess_cell_image(img)  # using the preprocess defined earlier
                cell_tensors.append(processed)
            cell_batch = torch.stack(cell_tensors).to(device)
            print("cell_batch.shape:", cell_batch.shape)
            try:
                outputs = resnet_model(cell_batch)
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            predictions_list.extend(predictions.cpu().numpy())
            confidences_list.extend(confidences.cpu().numpy())
            # Clean up memory
            del cell_batch, cell_tensors, outputs
            torch.cuda.empty_cache()
    return np.array(predictions_list), np.array(confidences_list)

# Preprocessing for ResNet model
def preprocess_cell_image(cell_image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
    ])
    return preprocess(cell_image)

def load_yolo_model(device):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.fuse()
    yolo_model.to(device)
    return yolo_model

def load_resnet_model(num_classes, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model

def create_checkbox_frame(master, row, column, padx=10, pady=10, sticky="nsew"):
    frame = ctk.CTkFrame(master)
    frame.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
    return frame

def create_path_input(master, width, row, column, padx=10, pady=10):
    path_input = ctk.CTkEntry(master, width=width)
    path_input.grid(row=row, column=column, padx=padx, pady=pady)
    return path_input

def create_button(master, text, row, column, command=None, padx=10, pady=10):
    button = ctk.CTkButton(master, text=text, command=command)
    button.grid(row=row, column=column, padx=padx, pady=pady)
    return button

def create_checkbox(master, text, command, row, column, padx=10, pady=10, state="normal", selected=False):
    checkbox = ctk.CTkCheckBox(master=master, text=text, command=command, state=state)
    checkbox.grid(row=row, column=column, padx=padx, pady=pady)
    if selected:
        checkbox.select()
    return checkbox

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

def log_to_console(console_box, message):
    console_box.configure(state="normal")
    console_box.insert("end", f"{message}\n")
    console_box.configure(state="disabled")
    console_box.yview("end")

def browse_dialog(dialog_type, filetypes=None):
    if dialog_type == "directory":
        return filedialog.askdirectory()
    elif dialog_type == "file":
        return filedialog.askopenfilename(filetypes=filetypes)
    return None

def browse_output_directory(isDir, pathDialog):
    if isDir.get():
        selected_path = browse_dialog("directory")
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

def file_input_criteria_toggle(isDir, isFile, dirCheckbox, fileCheckbox):
    fileCheckbox.configure(state="disabled" if isDir.get() else "normal")
    dirCheckbox.configure(state="disabled" if isFile.get() else "normal")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("V1")
        self.geometry("1100x580")
        self.resizable(False, False)

        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # UI
        self._create_sidebar()
        self.active_screen = None
        print("Current thread (App):", threading.current_thread())

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

class SVS_TO_JPG_SCREEN(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)
        self.configure(fg_color="#2c2f33")  # Background color

        # Title
        self.title_label = ctk.CTkLabel(self, text="SVS to JPG Converter", font=("Arial", 20, "bold"))
        self.title_label.grid(row=0, column=1, padx=60, pady=(20, 10), sticky="ew")

        # SVS Input Section
        self.svs_label = ctk.CTkLabel(self, text="SVS Input", font=("Arial", 14))
        self.svs_label.grid(row=1, column=0, padx=(120, 20), pady=(10, 5), sticky="w")

        self.svs_path = create_path_input(self, width=300, row=1, column=1)
        self.select_svs = create_button(self, text="Browse", row=1, column=2, command=self.browse_svs)
        self.checkbox_frame_kfb = create_checkbox_frame(self, row=2, column=1)
        self.svs_isDir = create_checkbox(master=self.checkbox_frame_kfb, text="isDir", command=self.svs_checkbox_toggle, row=0, column=0, selected=True)
        self.svs_isFile = create_checkbox(master=self.checkbox_frame_kfb, text="isFile", command=self.svs_checkbox_toggle, row=0, column=1)

        # JPG Output Section
        self.output_label = ctk.CTkLabel(self, text="JPG Output", font=("Arial", 14))
        self.output_label.grid(row=3, column=0, padx=(120, 20), pady=(10, 5), sticky="w")

        self.output_path = create_path_input(self, width=300, row=3, column=1)
        self.select_output = create_button(self, text="Browse", row=3, column=2, command=self.browse_output)
        self.checkbox_frame_output = create_checkbox_frame(self, row=4, column=1)
        self.output_isDir = create_checkbox(master=self.checkbox_frame_output, text="isDir", command=None, row=0, column=0, state="disabled", selected=True)

        # Console Output Area
        self.console_output = ctk.CTkTextbox(self, height=150, wrap="word", font=("Courier", 12))
        self.console_output.grid(row=5, column=0, columnspan=3, padx=(120, 10), pady=20, sticky="ew")
        self.console_output.configure(state="disabled")

        # Flag to keep track of start/stop events
        self.stop_event = threading.Event()

        # Start/Cancel Button
        self.start_stop_button = create_button(self, text="Start Conversion", row=6, column=2, command=self.start_stop_toggle)

        self.svs_checkbox_toggle()

    def svs_checkbox_toggle(self):
        file_input_criteria_toggle(self.svs_isDir, self.svs_isFile, self.svs_isDir, self.svs_isFile)

    def start_stop_toggle(self):
        if self.start_stop_button.cget("text") == "Start Conversion":
            self.start_conversion()
            if not hasattr(self, 'thread'):
                pass
            else: #if self.thread exist, only change state to cancel
                self.start_stop_button.configure(text="Cancel", fg_color="red", hover_color="#a83232")
        else:
            if not hasattr(self, 'thread'):
                return
            elif self.thread and self.thread.is_alive():
                self.cancel_conversion()
                self.start_stop_button.configure(text="Start Conversion", fg_color="#7289da", hover_color="#5b6eae")

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
        
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self.convert_svs_to_jpg_tiles_parallel,
            args=(svs_dir, output_dir, 1024, 0, 4),
            daemon=True
        )
        self.thread.start()

    def cancel_conversion(self):
        self.stop_event.set()  # Set stop flag

    def convert_svs_to_jpg_tiles_parallel(self, input_path, output_dir, tile_size=1024, level=0, max_workers=4):
        """
        Split an SVS file into multiple JPG tiles in parallel.

        Parameters:
        - input_path: Input SVS file path
        - output_dir: Directory for output JPG files
        - tile_size: Size of each tile (default is 1024x1024 pixels)
        - level: Image level to read (default 0 is the highest resolution)
        - max_workers: Number of threads for parallel processing (default is 4) [Disabled for UI responsiveness as of now]
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
                        if self.stop_event.is_set():
                            messagebox.showerror("Conversion Cancelled", "Conversion was cancelled.")
                            return
                        self.process_tile(image, scale, tx, ty, tile_size, level, file_output_dir)

            messagebox.showinfo("Success", "Conversion complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {e}")

        self.start_stop_button.configure(text="Start Conversion", fg_color="#7289da", hover_color="#5b6eae")

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
            log_to_console(self.console_output, f"Saved tile: {output_path}")
        except Exception as e:
            log_to_console(self.console_output, f"Error processing tile ({tx}, {ty}): {e}")

class analyze_JPG_SCREEN(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=960, height=540, corner_radius=10)
        self.configure(fg_color="#2c2f33")

        # Title (row 0)
        self.title_label = ctk.CTkLabel(self, text="Analyze Images", font=("Arial", 20, "bold"))
        self.title_label.grid(row=0, column=1, padx=60, pady=(20, 10), sticky="ew")

        # JPG Input Section (row 1)
        self.jpg_label = ctk.CTkLabel(self, text="JPG Folder", font=("Arial", 14))
        self.jpg_label.grid(row=1, column=0, padx=(120, 20), pady=(10, 5), sticky="w")
        self.jpg_path = create_path_input(self, width=300, row=1, column=1)
        self.select_jpg = create_button(self, text="Browse", row=1, column=2, command=self.browse_jpg)

        # Output Section (row 2)
        self.output_label = ctk.CTkLabel(self, text="AI Output", font=("Arial", 14))
        self.output_label.grid(row=2, column=0, padx=(120, 20), pady=(10, 5), sticky="w")
        self.output_path = create_path_input(self, width=300, row=2, column=1)
        self.select_output = create_button(self, text="Browse", row=2, column=2, command=self.browse_output)

        # (Optional) Remove the output_isDir checkbox if you no longer need it
        # Or, if you still need it, adjust its row accordingly.
        self.checkbox_frame_output = create_checkbox_frame(self, row=3, column=1)
        self.output_isDir = create_checkbox(master=self.checkbox_frame_output, text="isDir",
                                            command=None, row=0, column=0, state="disabled", selected=True)

        # Hidden Variable (not displayed)
        self.jpg_isDir = create_checkbox(master=None, text="isDir", command=None, row=0, column=0, selected=True)
        self.jpg_isDir.grid_forget()

        # Console Output Area (row 4)
        self.console_output = ctk.CTkTextbox(self, height=150, wrap="word", font=("Courier", 12))
        self.console_output.grid(row=4, column=0, columnspan=3, padx=(120, 10), pady=20, sticky="ew")
        self.console_output.configure(state="disabled")
        log_to_console(self.console_output, "If GPU can't be enabled, it means CUDA isn't detected in system.")

        # Flag to keep track of start/stop events
        self.stop_event = threading.Event()

        # Start/Cancel Button (row 5)
        self.start_stop_button = create_button(self, text="Start Analyzing", row=5, column=2, command=self.start_stop_toggle)

        # Select CPU as default first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device_checkbox_toggle()

    def device_checkbox_toggle(self):
        self.isGPU.configure(state="disabled" if not torch.cuda.is_available() or self.isCPU.get() else "normal")
        self.isCPU.configure(state="disabled" if self.isGPU.get() else "normal")

    def browse_jpg(self):
        browse_input(self.jpg_isDir, None, self.jpg_path, None)

    def browse_output(self):
        browse_output_directory(self.output_isDir, self.output_path)

    def start_stop_toggle(self):
        if self.start_stop_button.cget("text") == "Start Analyzing":
            self.start_multiprocessing_analyze()
            self.start_stop_button.configure(text="Cancel", fg_color="red", hover_color="#a83232")
        else:
            self.cancel_analyze()
            self.start_stop_button.configure(text="Start Analyzing", fg_color="#7289da", hover_color="#5b6eae")

    def start_multiprocessing_analyze(self):
        jpg_dir = self.jpg_path.get()
        output_dir = self.output_path.get()

        if not jpg_dir or not output_dir:
            messagebox.showerror("Error", "Empty directory is not allowed!")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create a multiprocessing Queue and Event to communicate with the process.
        self.queue = Queue()
        self.proc_stop_event = Event()

        # Start the heavy analysis process
        self.process = Process(target=process_analyze, args=(jpg_dir, output_dir, self.device, self.queue, self.proc_stop_event))
        self.process.start()

        # Start polling the queue on the main thread to update the UI.
        self.after(100, self.poll_queue)

    def cancel_analyze(self):
        if hasattr(self, 'proc_stop_event'):
            self.proc_stop_event.set()
        if hasattr(self, 'process'):
            self.process.join()

    def poll_queue(self):
        try:
            while not self.queue.empty():
                msg = self.queue.get_nowait()
                log_to_console(self.console_output, msg)
        except Exception as e:
            log_to_console(self.console_output, f"Error polling queue: {e}")
        # Continue polling
        self.after(100, self.poll_queue)

if __name__ == '__main__':
    app = App()
    app.mainloop()
