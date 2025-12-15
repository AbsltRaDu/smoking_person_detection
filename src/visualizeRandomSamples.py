import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy
import torch

import random


def visualize_random_samples(dataset, n=8, figsize=(14, 14)):
    """
    Показывает n случайных изображений из любого YOLO датасета.
    Использует dataset[i] -> (img, boxes)
    """
    idxs = random.sample(range(len(dataset)), n)

    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=figsize)

    for i, idx in enumerate(idxs, 1):
        img, boxes = dataset[idx]

        # Преобразуем изображение в numpy
        if torch.is_tensor(img):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)

        h, w = img_np.shape[:2]

        ax = plt.subplot(rows, cols, i)
        ax.imshow(img_np)
        ax.set_title(f"idx={idx}, boxes={len(boxes)}")
        ax.axis("off")

        # рисуем боксы
        for (cx, cy, bw, bh) in boxes:
            cx *= w
            cy *= h
            bw *= w
            bh *= h

            x1 = cx - bw / 2
            y1 = cy - bh / 2

            rect = patches.Rectangle(
                (x1, y1),
                bw,
                bh,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()