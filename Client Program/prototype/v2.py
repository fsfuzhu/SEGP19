import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F
import pyvips
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
mp.set_start_method('spawn', force=True)

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

RESNET_MODEL_PATH = './models/resnet18_model_20250301_epoch_16.pth'
YOLO_MODEL_PATH = './models/yolov8x_model.pt'

# ----To install SAM, please use this command: pip install git+https://github.com/facebookresearch/segment-anything.git\
# --- To download the model from this link: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
SAM_MODEL_PATH = './models/sam_vit_h_4b8939.pth'

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

def browse_input_directory(pathDialog):
    if True: # if statement simplified
        selected_path = browse_dialog("directory")
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

def browse_output_directory(pathDialog):
    selected_path = browse_dialog("directory")
    if selected_path:
        pathDialog.delete(0, tk.END)
        pathDialog.insert(0, selected_path)

def test_svs_tiles(svs_dir, output_dir, cell_detection_model):
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    resnet_model = load_resnet_model(RESNET_MODEL_PATH)
    sam_model = load_sam_model(SAM_MODEL_PATH)
    # 创建各分类输出目录
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    for file in os.listdir(svs_dir):
        if file.lower().endswith('.svs'):
            svs_path = os.path.join(svs_dir, file)
            if cell_detection_model == "sam":
                process_svs_file(svs_path, resnet_model, output_dir, sam_model=sam_model)
                print("hi sam")
            elif cell_detection_model == "yolo":
                process_svs_file(svs_path, resnet_model, output_dir, yolo_model=yolo_model)
                print("hi yolo")
            else:
                return
            
def process_svs_file(svs_path, resnet_model, output_dir, yolo_model = None, sam_model = None, tile_size=TILE_SIZE, detection_level=DETECTION_LEVEL):
    print(f"Processing SVS file: {svs_path}")
    try:
        full_slide = pyvips.Image.new_from_file(svs_path, access='sequential')
    except Exception as e:
        print(f"Failed to open SVS file: {e}")
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
    # 遍历所有瓷砖（此处采用顺序处理，也可使用线程池并行处理）
    for ty in range(tiles_y):
        for tx in range(tiles_x):
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
                continue
            process_tile(tile, tile_origin_x, tile_origin_y, scale, full_width, full_height, full_slide, resnet_model, output_dir, yolo_model=yolo_model, sam_model=sam_model)

def process_tile(tile, tile_origin_x, tile_origin_y, scale, full_width, full_height, full_slide, resnet_model, output_dir, yolo_model=None, sam_model=None):
    # tile 为经过 pyvips.crop() 并 resize 后的瓷砖，尺寸约为 TILE_SIZE×TILE_SIZE（检测级别下）
    tile_np = pyvips_to_numpy(tile)
    # 在瓷砖上运行 YOLO 检测
    try:
        if yolo_model:
            print("using yolo for detection")
            detections = yolo_detect_cells(yolo_model, tile_np)
        elif sam_model:
            print("using sam for detection")
            masks = sam_model.generate(tile_np)
            detections = sam_detect_cells(tile_np, masks)
    except Exception as e:
        print(f"Failed to detect cells: {e}")
        return

    if not detections:
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
        cell_np = pyvips_to_numpy(cell_region)
        preds, confs = classify_cells(resnet_model, [cell_np])
        class_idx = preds[0]
        conf = confs[0]
        class_name = CLASS_NAMES[class_idx]
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(cell_np, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLASS_COLOURS.get(class_name, (255,255,255)), 2)
        out_folder = os.path.join(output_dir, class_name)
        os.makedirs(out_folder, exist_ok=True)
        out_filename = f"tile_{tile_origin_x}_{tile_origin_y}_cell_{idx}.jpg"
        out_path = os.path.join(out_folder, out_filename)
        cv2.imwrite(out_path, cell_np)
        print(f"Saved cell image: {out_path}")

def pyvips_to_numpy(vimage):
    img = vimage.write_to_memory()
    arr = np.frombuffer(img, dtype=np.uint8)
    arr = arr.reshape(vimage.height, vimage.width, vimage.bands)
    if vimage.bands >= 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def yolo_detect_cells(yolo_model, image):
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

def sam_detect_cells(image, anns, padding=0):
    """
    Given an image and a list of annotations, this function will return a list of bounding boxes
    for each annotation that is not completely contained within another annotation.
    
    Args:
        image: The image as a numpy array.
        anns: A list of annotations, where each annotation is a dictionary with a 'segmentation' key
              that contains a binary mask.
        padding: The number of pixels to pad around each bounding box.
        
    Returns:
        A list of bounding boxes, where each bounding box is a list of four integers: [x_min, y_min, x_max, y_max].
    """
    # First, compute bounding boxes for each annotation.
    sorted_anns = sorted(anns, key=lambda x: x['area'])
    filtered_anns = sorted_anns
    filtered_anns = [sorted_anns[-1]] # Append the last (largest) mask first, since no other mask can be nested inside it

    for i, ann in enumerate(sorted_anns[:-1]):  # Iterate up to second-to-last
        nested = False
        bbox_i = ann['segmentation']
        for j in range(i+1, len(sorted_anns)):
            bbox_j = sorted_anns[j]['segmentation']

            # Nested mask check using XOR
            if np.all(np.logical_and(bbox_i, np.logical_not(bbox_j)) == 0):  # Check for all zeros
                nested = True
                break
        if not nested:
            filtered_anns.append(ann)

    image_size = image.shape[0] * image.shape[1]
    boxes = []  # Each element will be ((x_min, y_min, x_max, y_max))
    for ann in filtered_anns:
        mask = ann['segmentation']
        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Apply padding and ensure coordinates remain within image bounds.
        y_min = max(y_min - padding, 0)
        x_min = max(x_min - padding, 0)
        y_max = min(y_max + padding, image.shape[0] - 1)
        x_max = min(x_max + padding, image.shape[1] - 1)

        area = (y_max - y_min) * (x_max - x_min)
        if (area/image_size) < 1/8:  
            boxes.append((x_min, y_min, x_max, y_max))
    
    # Define a helper function to check if one box is contained within another.
    def is_contained(inner, outer):
        ix_min, iy_min, ix_max, iy_max = inner
        ox_min, oy_min, ox_max, oy_max = outer
        return (ix_min >= ox_min) and (iy_min >= oy_min) and (ix_max <= ox_max) and (iy_max <= oy_max)
    
    # Now filter out nested boxes:
    # We'll only draw a box if it is not completely contained within any other box.
    final_boxes = []
    for i, box_i in enumerate(boxes):
        nested = False
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            # If box_i is completely inside box_j, mark it as nested.
            if is_contained(box_i, box_j):
                nested = True
                break
        if not nested:
            x1, y1, x2, y2 = box_i
            final_boxes.append({'bbox': [x1, y1, x2, y2]})
    return final_boxes

def classify_cells(resnet_model, cell_images):
    cell_tensors = []
    for cell_img in cell_images:
        processed = preprocess_cell_image(cell_img)
        cell_tensors.append(processed)
    cell_batch = torch.stack(cell_tensors).to(device)
    with torch.no_grad():
        outputs = resnet_model(cell_batch)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)
    return predictions.cpu().numpy(), confidences.cpu().numpy()

def preprocess_cell_image(cell_image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    return preprocess(cell_image)

def load_yolo_model(model_path):
    yolo_model = YOLO(model_path)
    return yolo_model

def load_sam_model(sam_checkpoint, model_type="vit_h"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh = 0.7,
        stability_score_thresh = 0.95
    )
    return mask_generator

def load_resnet_model(model_path, num_classes=2):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

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
        self.select_svs_path = create_button(self, text="Browse SVS", row=0, column=0, padx=40, pady=20, command=self.browse_svs)
        self.svs_path = create_path_input(self, width=280, row=0, column=1)

        # Browse JPG Output (row 1)
        self.select_output_path = create_button(self, text="JPG Output", row=1, column=0, padx=40, pady=20, command=self.browse_output)
        self.jpg_path = create_path_input(self, width=280, row=1, column=1)

        # Select Basic/Advanced Detection (row 2)
        self.radio_var = tk.StringVar(value="yolo")
        self.radiobutton_frame = create_ctk_frame(self, row=2, column=0, columnspan=2, padx=(40, 10), sticky="ew")
        self.label_radio_group = ctk.CTkLabel(master=self.radiobutton_frame, text="Model Detection: ", font=("Calibri", 15))
        self.label_radio_group.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.basic_detect = create_radiobutton(master=self.radiobutton_frame, text="Basic", row=0, column=1, variable=self.radio_var, value="yolo")
        self.advance_detect = create_radiobutton(master=self.radiobutton_frame, text="Advanced", row=0, column=2,variable=self.radio_var, value="sam")

        # Start/Cancel Button (row 3)
        self.start_stop_button = ctk.CTkButton(self, text="Start", command=self.start_stop_toggle, font=("Calibri", 50))
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
        browse_input_directory(self.svs_path)

    def browse_output(self):
        browse_output_directory(self.jpg_path)

    def start_stop_toggle(self):
        print(self.radio_var.get())
        if self.start_stop_button.cget("text") == "Start":
            self.start_multiprocessing_analyze()
            self.start_stop_button.configure(text="Cancel", fg_color="red", hover_color="#a83232")
        else:
            self.cancel_analyze()
            self.start_stop_button.configure(text="Start", fg_color="#7289da", hover_color="#5b6eae")

    def start_multiprocessing_analyze(self):
        svs_dir = self.svs_path.get()
        output_dir = self.jpg_path.get()

        if not svs_dir or not output_dir:
            messagebox.showerror("Error", "Empty directory is not allowed!")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create a multiprocessing Queue and Event to communicate with the process.
        self.queue = Queue()
        self.proc_stop_event = Event()

        # Start the heavy analysis process
        self.process = Process(target=test_svs_tiles, args=(svs_dir, output_dir, self.radio_var.get()))
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
                # log_to_console(self.console_output, msg)
                print(msg)
        except Exception as e:
            # log_to_console(self.console_output, f"Error polling queue: {e}")
            print(f"Error polling queue: {e}")
        # Continue polling
        self.after(100, self.poll_queue)

if __name__ == "__main__":
    app = App()
    app.mainloop()