"""
outlier.py
Loại bỏ các bounding box có tâm lệch xa trung bình (outlier) trước khi lưu kết quả.
"""

from ultralytics import YOLO
import os
import torch
import numpy as np

def filter_outliers(boxes, k=1.5):
    """
    Loại bỏ các box có tâm (cx, cy) lệch xa trung bình.
    boxes: danh sách [(conf, x1, y1, x2, y2)]
    k: hệ số độ lệch chuẩn để xác định outlier.
    """
    if len(boxes) <= 2:
        return boxes  # nếu quá ít box thì không cần lọc

    centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for _, x1, y1, x2, y2 in boxes])
    mean_center = centers.mean(axis=0)
    dists = np.linalg.norm(centers - mean_center, axis=1)
    mean_dist = dists.mean()
    std_dist = dists.std()

    keep_idx = np.where(dists <= mean_dist + k * std_dist)[0]
    return [boxes[i] for i in keep_idx]


def main():
    # ------------------------
    # Config
    # ------------------------
    model_path = "./runs/detect/alldata_9m_top7/weights/best.pt"
    data_dir = "../AICUP25/test/images/"
    save_txt = "../AICUP25/test/9m001_top8_outlier.txt"
    conf_thres = 0.001   # Ngưỡng confidence thấp nhất
    max_boxes = 8        # Giới hạn số box sau lọc
    outlier_k = 1.5      # Hệ số lọc outlier
    device = 0 if torch.cuda.is_available() else "cpu"

    # ------------------------
    # Load model
    # ------------------------
    model = YOLO(model_path)

    # ------------------------
    # Prepare image list
    # ------------------------
    image_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    image_files.sort()

    # ------------------------
    # Run inference
    # ------------------------
    with open(save_txt, "w") as f:
        for img_path in image_files:
            results = model(img_path, conf=conf_thres, device=device)
            r = results[0]
            boxes = r.boxes

            if boxes is None or len(boxes) == 0:
                continue

            img_name = os.path.splitext(os.path.basename(r.path))[0]

            # Trích xuất thông tin box
            box_data = [
                (float(box.conf.item()), *box.xyxy[0].cpu().numpy())
                for box in boxes
            ]

            # Lọc outlier dựa theo tâm
            filtered_boxes = filter_outliers(box_data, k=outlier_k)

            # Sắp xếp theo confidence và giới hạn số lượng
            filtered_boxes.sort(key=lambda b: b[0], reverse=True)
            filtered_boxes = filtered_boxes[:max_boxes]

            for conf, x1, y1, x2, y2 in filtered_boxes:
                cls_id = int(r.boxes[0].cls.item())  # chỉ 1 lớp
                line = f"{img_name}\t{cls_id}\t{conf:.4f}\t{int(x1)}\t{int(y1)}\t{int(x2)}\t{int(y2)}"
                print(line)
                f.write(line + "\n")

    print(f"\n[Done] Saved filtered results to {save_txt}")


if __name__ == "__main__":
    main()
