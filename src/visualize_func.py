import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_boxes_simple(image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor,
                      mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225]):
    """
    image  : Tensor [3, H, W]
    boxes  : Tensor [N, 4] (xyxy)
    labels : Tensor [N]
    """

    img = image.detach().cpu()
    
    if img.min() < 0:    # значит, скорее всего изображение нормализовано
        mean = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        img = img * std + mean   # de-normalize
    
    boxes = boxes.detach().cpu()
    labels = labels.detach().cpu()

    # Tensor -> numpy (H, W, 3)
    img = img.permute(1, 2, 0).numpy()
    img = img.clip(0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 2,
            f"{int(label)}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.7, edgecolor="none")
        )

    plt.show()
