# 26th, Sep

import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO

# 标注文件路径
ann_file = "annotations/instances_train.json"
img_dir = "images/train"
save_dir = "output/cropped_objects"

os.makedirs(save_dir, exist_ok=True)

# 初始化 COCO API
coco = COCO(ann_file)

# 你要提取的类别，比如 "person"
category_name = "person"
cat_ids = coco.getCatIds(catNms=[category_name])

# 遍历所有目标
for ann_id in coco.getAnnIds(catIds=cat_ids):
    ann = coco.loadAnns(ann_id)[0]
    img_info = coco.loadImgs(ann["image_id"])[0]

    img_path = os.path.join(img_dir, img_info["file_name"])
    image = cv2.imread(img_path)
    if image is None:
        continue

    # 解码 mask
    mask = coco.annToMask(ann)

    # 抠图
    cropped = cv2.bitwise_and(image, image, mask=mask)

    # 根据 bbox 裁剪（可选）
    x, y, w, h = map(int, ann["bbox"])
    cropped = cropped[y:y+h, x:x+w]

    # 保存
    save_path = os.path.join(save_dir, f"{ann_id}_{img_info['file_name']}")
    cv2.imwrite(save_path, cropped)

print("完成！")
