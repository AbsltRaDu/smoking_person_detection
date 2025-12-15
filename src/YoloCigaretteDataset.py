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
    def __init__(self, img_dir, label_dir, img_files, transform=None, keep_class=0):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = img_files
        self.transform = transform
        self.keep_class = keep_class

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
        return img, boxes
