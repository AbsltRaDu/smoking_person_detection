import os

import torch
import torch.utils.data as data
from PIL import Image

def get_pos_neg_files(img_dir, label_dir, keep_class=0):
    """Возвращает два списка имён файлов (jpg/png) — pos (есть класс keep_class) и neg (нет)."""
    pos, neg = [], []
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)
        if not os.path.exists(txt_path):
            # если .txt нет — считаем как негатив (нет сигареты)
            neg.append(img_name)
            continue

        has_cls = False
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                except:
                    continue
                if cls == keep_class:
                    has_cls = True
                    break

        if has_cls:
            pos.append(img_name)
        else:
            neg.append(img_name)
    return pos, neg

class YoloCigaretteDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, img_files, transform=None, keep_class=0, RCNN=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = img_files
        self.transform = transform
        self.keep_class = keep_class
        self.RCNN = RCNN

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    if cls == self.keep_class:
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx, cy, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
            
        if self.RCNN:
            H, W = img.shape[-2], img.shape[-1]
            
            boxes = {
                'boxes': yolo_to_xyxy_pixels(boxes, H, W),
                'labels': torch.ones(len(boxes), dtype=torch.int64)
            }
        
        return img, boxes

def yolo_to_xyxy_pixels(boxes_cxcywh_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
   
    boxes = boxes_cxcywh_norm.float()
    
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    
    cx, cy, bw, bh = boxes.unbind(dim=1)

    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H

    xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # на всякий случай обрежем в пределах изображения
    xyxy[:, 0].clamp_(0, W - 1)
    xyxy[:, 2].clamp_(0, W - 1)
    xyxy[:, 1].clamp_(0, H - 1)
    xyxy[:, 3].clamp_(0, H - 1)

    
    
    return xyxy