# 26th, Sep

import os
import cv2
import numpy as np

# 输入输出路径
img_dir = "images"         # 存放原图的目录
label_dir = "labels"       # 存放 YOLO segmentation txt 的目录
save_dir = "output/cropped_objects"
os.makedirs(save_dir, exist_ok=True)

# 你要提取的类别
target_class = 0   # 比如 "person" 对应的 class_id=0

for file in os.listdir(img_dir):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    
    img_path = os.path.join(img_dir, file)
    label_path = os.path.join(label_dir, file.rsplit(".", 1)[0] + ".txt")
    
    if not os.path.exists(label_path):
        continue
    
    # 读取图像
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])
        
        if class_id != target_class:
            continue
        
        # 归一化坐标 → 像素坐标
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        coords[:, 0] = coords[:, 0] * w
        coords[:, 1] = coords[:, 1] * h
        polygon = coords.astype(np.int32)

        # 生成 mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # 抠图
        cropped = cv2.bitwise_and(img, img, mask=mask)

        # 根据 polygon 的外接矩形裁剪（可选）
        x, y, ww, hh = cv2.boundingRect(polygon)
        cropped = cropped[y:y+hh, x:x+ww]

        # 保存
        save_path = os.path.join(save_dir, f"{file.rsplit('.',1)[0]}_{idx}.png")
        cv2.imwrite(save_path, cropped)

print("完成！")