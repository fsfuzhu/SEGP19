import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F
import pyvips
import threading

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

RESNET_MODEL_PATH = './models/resnet18_model_20250301_epoch_16.pth'
YOLO_MODEL_PATH = './models/yolov8x_model.pt'

# 检测阈值
YOLO_CONFIDENCE_THRESHOLD = 0.5  # 根据实际情况调整
RESNET_CONFIDENCE_THRESHOLD = 0.0

# 细胞类别及标注颜色
CLASS_NAMES = ['abnormal', 'normal']
CLASS_COLOURS = {
    "normal": (100, 131, 54),
    "abnormal": (28, 32, 190),
}

# 瓷砖（tile）参数：
TILE_SIZE = 1024           # 在检测级别下，每个瓷砖尺寸（单位像素）
DETECTION_LEVEL = 1        # 用于检测的图像级别（一般低于 0 级可大幅降低尺寸）
EDGE_MARGIN = 20           # 瓷砖边缘判定阈值（单位：瓷砖图像像素）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Progress Bar
CURRENT_SVS_TOTAL_TILES = 0
PROCESSED_CURRENT_SVS_TILES = 0
TOTAL_SVS_FILE_COUNT = 0
PROCESSED_SVS_FILE_COUNT = 0

# UI Related functions
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

def browse_dialog(dialog_type, dialog_title, filetypes=None):
    if dialog_type == "directory":
        return filedialog.askdirectory(title=dialog_title)
    elif dialog_type == "file":
        return filedialog.askopenfilename(filetypes=filetypes, title=dialog_title)
    return None

def browse_directory(pathDialog, dialogTitle):
    selected_path = browse_dialog("directory", dialogTitle)
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

def load_yolo_model(model_path):
    yolo_model = YOLO(model_path)
    return yolo_model

def load_resnet_model(model_path, num_classes=2):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def get_current_svs_total_tiles():
    return CURRENT_SVS_TOTAL_TILES

def get_processed_current_svs_tiles():
    return PROCESSED_CURRENT_SVS_TILES

def get_total_svs_file_count():
    return TOTAL_SVS_FILE_COUNT

def get_processed_svs_file_count():
    return PROCESSED_SVS_FILE_COUNT

def set_current_svs_total_tiles(value):
    global CURRENT_SVS_TOTAL_TILES
    CURRENT_SVS_TOTAL_TILES = value

def set_processed_current_svs_tiles(value):
    global PROCESSED_CURRENT_SVS_TILES
    PROCESSED_CURRENT_SVS_TILES = value

def set_total_svs_file_count(value):
    global TOTAL_SVS_FILE_COUNT
    TOTAL_SVS_FILE_COUNT = value

def set_processed_svs_file_count(value):
    global PROCESSED_SVS_FILE_COUNT
    PROCESSED_SVS_FILE_COUNT = value

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("Pap Smear Analysis Tool")
        self.center_window(552, 769)
        self.resizable(False, False)

        # Grid config
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(7, weight=1)

        # SVS Input Section (row 0)
        self.select_svs_path = create_button(self, text="SVS Folder", row=0, column=0, padx=40, pady=20, command=self.browse_svs)
        self.svs_path = create_path_input(self, width=280, row=0, column=1)

        # Browse JPG Output (row 1)
        self.select_output_path = create_button(self, text="Cell Output Folder", row=1, column=0, padx=40, pady=20, command=self.browse_output)
        self.jpg_path = create_path_input(self, width=280, row=1, column=1)

        # Save/Discard Images (row 2)
        self.radio_var = tk.BooleanVar(value=True)
        self.radiobutton_frame = create_ctk_frame(self, row=2, column=0, columnspan=2, padx=(40, 10), sticky="ew")
        self.label_radio_group = ctk.CTkLabel(master=self.radiobutton_frame, text="Save Cell Images to Disk: ", font=("Calibri", 15))
        self.label_radio_group.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.save_image = create_radiobutton(master=self.radiobutton_frame, text="Yes", row=0, column=1, variable=self.radio_var, value=True)
        self.discard_image = create_radiobutton(master=self.radiobutton_frame, text="No", row=0, column=2,variable=self.radio_var, value=False)
        self.radio_var.trace_add("write", self.toggle_output_button)

        # Start/Cancel Button (row 3)
        self.start_stop_button = ctk.CTkButton(self, text="Start", command=self.start_stop_toggle, font=("Calibri", 50))
        self.start_stop_button.grid(row=3, column=0, columnspan=2, padx=(40, 10), pady=20, sticky="ew")
        
        # Progress Bar Frame Current (row 4)
        self.progress_frame_current = create_ctk_frame(self, row=4, column=0, columnspan=2, padx=(40, 10), sticky="ew")
        self.progress_label_current = ctk.CTkLabel(master=self.progress_frame_current, text="Current Progress:", font=("Calibri", 20))
        self.progress_label_current.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Progress Bar (Current)
        self.progress_bar_current = ctk.CTkProgressBar(master=self.progress_frame_current)
        self.progress_bar_current.grid(row=0, column=1, padx=(10, 10), pady=10, sticky="e")
        self.progress_frame_current.grid_columnconfigure(1, weight=1)

        # Percentage Label (Current)
        self.percentage_label_current = ctk.CTkLabel(master=self.progress_frame_current, text="0.0%", font=("Calibri", 20))
        self.percentage_label_current.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

        self.progress_bar_current.set(0.0)

        # Progress Bar Frame Total (row 5)
        self.progress_frame_total = create_ctk_frame(self, row=5, column=0, columnspan=2, padx=(40, 10), sticky="ew")
        self.progress_label_total = ctk.CTkLabel(master=self.progress_frame_total, text="Total Progress:", font=("Calibri", 20))
        self.progress_label_total.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Progress Bar (Total)
        self.progress_bar_total = ctk.CTkProgressBar(master=self.progress_frame_total)
        self.progress_bar_total.grid(row=0, column=1, padx=(10, 10), pady=10, sticky="e")
        self.progress_frame_total.grid_columnconfigure(1, weight=1)

        # Percentage Label (Total)
        self.percentage_label_total = ctk.CTkLabel(master=self.progress_frame_total, text="0.0%", font=("Calibri", 20))
        self.percentage_label_total.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

        self.progress_bar_total.set(0.0)

        # Data Table (For Analytical Review)
        self.data_frame = ctk.CTkFrame(self)
        self.data_frame.grid(row=6, column=0, columnspan=2, padx=(40, 10), pady=10, sticky="ns")
        self.tree = ttk.Treeview(self.data_frame, columns=("File Name", "Cell Type", "Cell Count"), show='headings')

        # Configure table headers
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor="center")

        # Add scrollbars for improved navigation
        self.scroll_y = ttk.Scrollbar(self.data_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=self.scroll_y.set)

        # Grid placement for table and scrollbar
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        # Enable resizing for dynamic size
        self.data_frame.grid_rowconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(0, weight=1)

        # Flag to keep track of start/stop events
        self.stop_event = threading.Event()

    def center_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def browse_svs(self):
        browse_directory(self.svs_path, "Select SVS Folder")

    def browse_output(self):
        browse_directory(self.jpg_path, "Select Output Folder")

    def toggle_output_button(self, *args):
        if self.radio_var.get():
            self.select_output_path.configure(state="normal")
        else:
            self.select_output_path.configure(state="disabled")

    def start_stop_toggle(self):
        if self.start_stop_button.cget("text") == "Start":
            self.start_multiprocessing_analyze()
            if not hasattr(self, 'thread'):
                pass
            else: #if self.thread exist, only change state to cancel
                self.start_stop_button.configure(text="Cancel", fg_color="red", hover_color="#a83232")
        else:
            if not hasattr(self, 'thread'):
                return
            elif self.thread and self.thread.is_alive():
                self.cancel_analyze()
                self.start_stop_button.configure(text="Start", fg_color="#7289da", hover_color="#5b6eae")

    def start_multiprocessing_analyze(self):
        svs_dir = self.svs_path.get()
        output_dir = self.jpg_path.get()

        if not svs_dir or (self.radio_var.get() and not output_dir):
            messagebox.showerror("Error", "Empty directory is not allowed!")
            self.cancel_analyze()
            return
        
        if not os.path.exists(output_dir) and self.radio_var.get():
            os.makedirs(output_dir, exist_ok=True)

        self.stop_event.clear()

        self.thread = threading.Thread(
            target = self.test_svs_tiles,
            args = (svs_dir, output_dir),
            daemon = True
        )
        self.thread.start()
        self.update_progress_bar(reset_flag=True)

    def cancel_analyze(self):
        self.stop_event.set()  # Set stop flag

    def cancel_task(self):
        messagebox.showerror("Analyzing Cancelled", "Processing of images has been cancelled.")
        self.update_progress_bar(reset_flag=True)

    def update_progress_bar(self, reset_flag=False):
        if reset_flag:
            self.progress_bar_current.set(0.0)
            self.progress_bar_total.set(0.0)
            self.percentage_label_current.configure(text="0.0%")
            self.percentage_label_total.configure(text="0.0%")
            set_current_svs_total_tiles(0)
            set_processed_current_svs_tiles(0)
            set_processed_svs_file_count(0)
            set_total_svs_file_count(0)
        else:
            current_progress = get_processed_current_svs_tiles() / get_current_svs_total_tiles()
            total_progress = (current_progress + (get_processed_svs_file_count() * 1)) / get_total_svs_file_count()
            self.progress_bar_current.set(current_progress)
            self.progress_bar_total.set(total_progress)
            self.percentage_label_current.configure(text=f"{current_progress*100:.1f}%")
            self.percentage_label_total.configure(text=f"{total_progress*100:.1f}%")

    def populate_data_table(self, dict):
        for file_name, counts in dict.items():
            total_cells = sum(counts.values())
            for cell_type, count in counts.items():
                percentage = (count / total_cells * 100) if total_cells > 0 else 0
                self.tree.insert("", "end", values=(file_name, cell_type.capitalize(), f"{count} ({percentage:.2f}%)"))

    # Logic
    def test_svs_tiles(self, svs_dir, output_dir):
        self.cell_counts = {}
        yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        resnet_model = load_resnet_model(RESNET_MODEL_PATH)
        set_total_svs_file_count(sum(file.lower().endswith('.svs') for file in os.listdir(svs_dir)))
        for file in os.listdir(svs_dir):
            if file.lower().endswith('.svs'):
                svs_path = os.path.join(svs_dir, file)
                file_name = os.path.splitext(os.path.basename(file))[0]
                file_output_dir = os.path.join(output_dir, file_name)

                if self.radio_var.get():
                    for class_name in CLASS_NAMES:
                        os.makedirs(os.path.join(file_output_dir, class_name), exist_ok=True)

                set_current_svs_total_tiles(0)
                set_processed_current_svs_tiles(0)
                self.cell_counts[file_name] = {"normal": 0, "abnormal": 0}
                self.process_svs_file(svs_path, resnet_model, file_output_dir, file_name, yolo_model=yolo_model)
                set_processed_svs_file_count(get_processed_svs_file_count() + 1)
                self.tree.delete(*self.tree.get_children())
                self.populate_data_table(self.cell_counts)
                
        self.start_stop_button.configure(text="Start", fg_color="#7289da", hover_color="#5b6eae")

    def process_svs_file(self, svs_path, resnet_model, output_dir, file_name, yolo_model = None, tile_size=TILE_SIZE, detection_level=DETECTION_LEVEL):
        print(f"Processing SVS file: {svs_path}")
        
        try:
            full_slide = pyvips.Image.new_from_file(svs_path, access='sequential')
        except Exception as e:
            messagebox.showerror("SVS Error", f"Failed to open SVS file: {e}")
            self.cancel_analyze()
            return
        # 获取指定检测级别下的 downsample 因子（相对于 level0）
        try:
            scale_str = full_slide.get('openslide.level[{}].downsample'.format(detection_level))
            scale = float(scale_str)
        except Exception as e:
            print(f"Could not get downsample factor for level {detection_level}, defaulting to 1.0: {e}")
            scale = 1.0

        full_width = full_slide.width
        full_height = full_slide.height
        level_width = int(full_width / scale)
        level_height = int(full_height / scale)
        print(f"Detection level {detection_level} dimensions: {level_width} x {level_height}, scale: {scale}")
        tiles_x = (level_width + tile_size - 1) // tile_size
        tiles_y = (level_height + tile_size - 1) // tile_size
        print(f"Dividing slide into {tiles_x} x {tiles_y} = {tiles_x * tiles_y} tiles")
        set_current_svs_total_tiles(tiles_x * tiles_y)

        # 遍历所有瓷砖（此处采用顺序处理，也可使用线程池并行处理）
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                if self.stop_event.is_set():
                    self.cancel_task()
                    return
                tile_origin_x = tx * tile_size
                tile_origin_y = ty * tile_size
                x_full = int(tile_origin_x * scale)
                y_full = int(tile_origin_y * scale)
                w_full = int(tile_size * scale)
                h_full = int(tile_size * scale)
                try:
                    # 从全幅图像中裁剪瓷砖区域，然后利用 resize 将其还原为检测级别尺寸
                    tile = full_slide.crop(x_full, y_full, w_full, h_full).resize(1/scale)
                except Exception as e:
                    print(f"Failed to extract tile at ({tx}, {ty}): {e}")
                    set_current_svs_total_tiles(get_current_svs_total_tiles() - 1)
                    self.update_progress_bar()
                    continue
                self.process_tile(tile, tile_origin_x, tile_origin_y, scale, full_width, full_height, full_slide, resnet_model, file_name, output_dir, yolo_model=yolo_model)

    def process_tile(self, tile, tile_origin_x, tile_origin_y, scale, full_width, full_height, full_slide, resnet_model, file_name, output_dir, yolo_model=None):
        # tile 为经过 pyvips.crop() 并 resize 后的瓷砖，尺寸约为 TILE_SIZE×TILE_SIZE（检测级别下）
        tile_np = self.pyvips_to_numpy(tile)
        # 在瓷砖上运行 YOLO 检测
        try:
            detections = self.yolo_detect_cells(yolo_model, tile_np)
        except Exception as e:
            messagebox.showerror("Cell Detection Error", f"Failed to detect cells: {e}")
            return

        if not detections:
            set_processed_current_svs_tiles(get_processed_current_svs_tiles() + 1)
            self.update_progress_bar()
            return
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            # 判断检测框是否接近瓷砖边缘
            touches_edge = (x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or x2 > (TILE_SIZE - EDGE_MARGIN) or y2 > (TILE_SIZE - EDGE_MARGIN))
            # 将瓷砖内检测框坐标转换为全分辨率下的全局坐标（level0）：
            global_x1 = int((tile_origin_x + x1) * scale)
            global_y1 = int((tile_origin_y + y1) * scale)
            global_x2 = int((tile_origin_x + x2) * scale)
            global_y2 = int((tile_origin_y + y2) * scale)
            margin_full = int(EDGE_MARGIN * scale)
            # 若检测框在边缘，则以检测中心为基准重新确定裁剪区域
            if touches_edge:
                center_x = (global_x1 + global_x2) // 2
                center_y = (global_y1 + global_y2) // 2
                box_width = global_x2 - global_x1
                box_height = global_y2 - global_y1
                crop_width = box_width + 2 * margin_full
                crop_height = box_height + 2 * margin_full
                new_x1 = max(0, center_x - crop_width // 2)
                new_y1 = max(0, center_y - crop_height // 2)
                new_x2 = min(full_width, new_x1 + crop_width)
                new_y2 = min(full_height, new_y1 + crop_height)
            else:
                new_x1 = max(0, global_x1 - margin_full)
                new_y1 = max(0, global_y1 - margin_full)
                new_x2 = min(full_width, global_x2 + margin_full)
                new_y2 = min(full_height, global_y2 + margin_full)
            crop_w = new_x2 - new_x1
            crop_h = new_y2 - new_y1
            try:
                cell_region = full_slide.crop(new_x1, new_y1, crop_w, crop_h)
            except Exception as e:
                print(f"Failed to crop cell region: {e}")
                continue
            cell_np = self.pyvips_to_numpy(cell_region)
            preds, confs = self.classify_cells(resnet_model, [cell_np])
            class_idx = preds[0]
            conf = confs[0]
            if class_idx == 0:
                self.cell_counts[file_name]["abnormal"] += 1
            elif class_idx == 1:
                self.cell_counts[file_name]["normal"] += 1
            class_name = CLASS_NAMES[class_idx]
            label = f"{class_name}: {conf:.2f}"

            if (self.radio_var.get()):
                cv2.putText(cell_np, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLASS_COLOURS.get(class_name, (255,255,255)), 2)
                out_folder = os.path.join(output_dir, class_name)
                os.makedirs(out_folder, exist_ok=True)
                out_filename = f"tile_{tile_origin_x}_{tile_origin_y}_cell_{idx}.jpg"
                out_path = os.path.join(out_folder, out_filename)
                cv2.imwrite(out_path, cell_np)
                print(f"Saved cell image: {out_path}")

        set_processed_current_svs_tiles(get_processed_current_svs_tiles() + 1)
        self.update_progress_bar()

    def pyvips_to_numpy(self, vimage):
        img = vimage.write_to_memory()
        arr = np.frombuffer(img, dtype=np.uint8)
        arr = arr.reshape(vimage.height, vimage.width, vimage.bands)
        if vimage.bands >= 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def yolo_detect_cells(self, yolo_model, image):
        results = yolo_model.predict(source=image, save=False)
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
        return boxes

    def classify_cells(self, resnet_model, cell_images):
        cell_tensors = []
        for cell_img in cell_images:
            if self.stop_event.is_set():
                self.cancel_task()
                return
            processed = self.preprocess_cell_image(cell_img)
            cell_tensors.append(processed)
        cell_batch = torch.stack(cell_tensors).to(device)
        with torch.no_grad():
            outputs = resnet_model(cell_batch)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
        return predictions.cpu().numpy(), confidences.cpu().numpy()

    def preprocess_cell_image(self, cell_image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                std =[0.229,0.224,0.225])
        ])
        return preprocess(cell_image)

if __name__ == "__main__":
    MainApp().mainloop()