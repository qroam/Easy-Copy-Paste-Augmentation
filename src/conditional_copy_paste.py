# Created in 26th, June
# Modified in 10th, July  (changed into conditional copy paste)
# Refactored in 17th, Sep (based on the previous version ./create_data.py)
# Refactored and Documented in 22nd, Sep
# Refactored in 24th, Sep
# 26th, Sep 支持通过传入路径，而不是文件名，来制定前景物体
# 修改阅读yaml文件时encoding为 encoding="utf-8"，保证可以读取中文路径

"""
conda activate <your-env-name>
cd ..
    
python conditional_copy_paste.py \
    --work_dir ".\example" \
    --config_filename "foreign_object_v1.yaml" \
    --do_feather \
    --do_edge_blur \
    --output_num_per_img 10 \
    --copy_existing_annotations
"""

# TODO 
# 待解决的BUG
# 贴图后物体轮廓超出背景图片边界的问题 (√)
# 边缘羽化和亮度调整 (√)

import os
import random
import yaml
import json

import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
from pathlib import Path

import argparse

from utils import polygon_to_bbox, bbox_to_yolo, bbox_to_yolo_segmentation, bbox_to_4_points_polygon
from conditional_rule_solver import choose_position_scale_angle_depend_on_rule, load_data_annotation
from shapely_clip import clip_polygon_to_image
from blur.blur_and_scale import do_blur_and_scale
from blur.utils import pil_to_numpy, numpy_to_pil


"""
✅ 标签保存建议
若你未来想用分割模型（如 YOLOv8-seg, Mask R-CNN）：

polygon 多边形格式是兼容的

若你希望后续保存为 COCO 或 YOLO 格式，我可以帮你进一步转换


✅ 拓展建议
功能	方法
支持像素级 mask	构建一张 label mask，与图像等大，物体区域赋类 ID
支持 YOLO bbox 格式	cv2.boundingRect(new_points) 得到 (x,y,w,h)，再归一化
支持 Albumentations 统一变换	若需要对图像 + mask 统一操作，建议引入 Albumentations
"""

DEFAULT_MINIMUM_OBJ_NUM = 0
DEFAULT_MAXIMUM_OBJ_NUM = 2
SCALING_CONSTANT = 1700

BACKGROUND_IMG_FOLDER_NAME = "background_img"
BACKGROUND_ANNOTATION_FOLDER_NAME = "background_annotations"
OBJECT_IMG_FOLDER_NAME = "object_img"
OUTPUT_IMG_FOLDER_NAME = "images"
OUTPUT_JSON_ANNOTATION_FOLDER_NAME = "annotations"
OUTPUT_YOLO_BBOX_ANNOTATION_FOLDER_NAME = "labels"
OUTPUT_YOLO_SEG_ANNOTATION_FOLDER_NAME = "labels_segment"



def get_affine_matrix(center, scale: float, angle_deg: float) -> np.ndarray:
    """给定旋转、缩放操作的参数，得到仿射变换的操作矩阵
    """
    # PIL 旋转是逆时针，OpenCV 也是
    M = cv2.getRotationMatrix2D(center=center, angle=angle_deg, scale=scale)  # 2x3 仿射矩阵
    return M

def apply_affine_transform(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    对一组点(points)应用操作矩阵(M)所描述的仿射变换
    Args:
        points (np.ndarray): points：点集，形状通常是 (N, 2)，表示 N 个二维点 [x, y]。
        M (np.ndarray): M：仿射变换矩阵，形状 (2, 3)。
        在 OpenCV 中，这通常由 cv2.getRotationMatrix2D() 或 cv2.getAffineTransform() 得到，比如：
        [ a11  a12  b1 ]
        [ a21  a22  b2 ]
        它描述了旋转、缩放、平移、剪切等线性变换。
    Returns:
        np.ndarray. shaped (N, 2)
    """
    ones = np.ones((points.shape[0], 1))  # 构造齐次坐标. 原始点是 [x, y]，现在扩展为 [x, y, 1]，形状 (N, 3). 这是因为仿射变换矩阵是 2x3，需要额外的一维来表示平移
    points_aug = np.hstack([points, ones])  # [x, y, 1]
    transformed = points_aug @ M.T  # [x', y']
    return transformed


def get_rotation_offset(w, h, angle_deg):
    """
    计算图像旋转后 expand=True 导致的新图像左上角在原图坐标系中的偏移。
    即原图 (0,0) 变换后在新图像坐标中的位置。
    """
    angle = np.radians(angle_deg)
    cos = np.cos(angle)
    sin = np.sin(angle)

    # 原图中心
    cx, cy = w / 2, h / 2

    # 原图四个角 (相对于中心点的偏移向量)
    corners = np.array([
        [-cx, -cy],  # 左上
        [ cx, -cy],  # 右上
        [ cx,  cy],  # 右下
        [-cx,  cy],  # 左下
    ])

    # 对四个角做旋转变换
    rot_mat = np.array([[cos, -sin],
                        [sin,  cos]])
    rotated = corners @ rot_mat.T

    # 找到旋转后图像的最小 x/y，即左上角在新图像中的位置
    min_x = rotated[:,0].min()
    min_y = rotated[:,1].min()

    # 原点 (0,0) 会被平移到 (cx - min_x, cy - min_y)
    offset_x = -min_x
    offset_y = -min_y
    
    offset_x -= w / 2
    offset_y -= h / 2

    return offset_x, offset_y

def get_scaling_offset(w, h, scale):
    """弥补PIL体系和openCV体系对于图片缩放变化所采用的参考系不同而带来的坐标差值

    Args:
        w (_type_): 图像的宽度
        h (_type_): 图像的高度
        scale (_type_): 缩放系数

    Returns:
        _type_: _description_
    """
    return w / 2 * (scale - 1), h / 2 * (scale - 1)



def main(
    work_dir: str, 
    output_folder_name: str = "output", 
    config_filename: str = "data.yaml", 
    # annotation_mode: str = "segmentation",
    do_feather: bool = True,
    do_edge_blur: bool = True,
    do_brightness_scale: bool = True,
    feather_radius: int = 5,
    output_num_per_img: int = 1,
    copy_existing_annotations: bool = False,
):

    # ========== 确定实验类型（检测或分割） ==========
    # assert annotation_mode in ["detection", "segmentation"]
    
    # ========== 实验设置文件 ==========
    data_config_filename = os.path.join(work_dir, config_filename)
        
    # ========== 背景图片文件夹和背景图片标注（可选） ==========    
    background_img_folder = os.path.join(work_dir, BACKGROUND_IMG_FOLDER_NAME)
    # TODO
    background_annotations_folder = os.path.join(work_dir, BACKGROUND_ANNOTATION_FOLDER_NAME)
    
    # ========== 前景目标物体文件夹 ==========
    object_img_folder = os.path.join(work_dir, OBJECT_IMG_FOLDER_NAME)
    
    # ========== 创建输出文件夹 ==========
    output_folder_dir = os.path.join(work_dir, output_folder_name)
    output_img_dir = os.path.join(work_dir, output_folder_name, OUTPUT_IMG_FOLDER_NAME)
    output_ann_dir = os.path.join(work_dir, output_folder_name, OUTPUT_JSON_ANNOTATION_FOLDER_NAME)
    output_yolo_bbox_ann_dir = os.path.join(work_dir, output_folder_name, OUTPUT_YOLO_BBOX_ANNOTATION_FOLDER_NAME)
    output_yolo_seg_ann_dir = os.path.join(work_dir, output_folder_name, OUTPUT_YOLO_SEG_ANNOTATION_FOLDER_NAME)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)
    os.makedirs(output_yolo_bbox_ann_dir, exist_ok=True)
    os.makedirs(output_yolo_seg_ann_dir, exist_ok=True)
    
    # ========== 加载实验设置文件 ==========
    with open(data_config_filename, "r", encoding="utf-8") as f:
        object_dict = yaml.safe_load(f)
        print(f"已加载以下的物体类别贴图数据集：\n{object_dict}")
    # 读取限制条件（可选）
    # TODO
    data_creation_rules = None
    if "rules" in object_dict:
        data_creation_rules = object_dict.pop("rules")
        print("data_creation_rules", data_creation_rules)
    
    number_rules = {}
    if "number" in object_dict:
        number_rules = object_dict.pop("number")
        print("number_rules", number_rules)
        
    number_classes_copy_paste = len(object_dict)
    class_name_id_mapping = {
        cls_name: cls_id for cls_id, cls_name in enumerate(list(object_dict.keys()))
    }
    print(class_name_id_mapping)
    
    
    # ========== 确定背景图片文件名 ==========
    background_list = os.listdir(background_img_folder)
    background_list = [fn for fn in os.listdir(background_img_folder) if fn.endswith(".jpg")]
    print(f"现在对背景图片共{len(background_list)}张进行贴图数据合成")
    
    # ========== 根据是否有对应的标注文件对背景图片做过滤 ==========
    if os.path.exists(background_annotations_folder):
        background_annotation_file_list = os.listdir(background_annotations_folder)
        print(f"其中，{len(background_annotation_file_list)}张图片带有匹配的标注数据")
        
        # TODO 此处过滤掉了不含标注数据的背景图片，应考虑改为可选
        background_list = [fn for fn in background_list if fn.replace(".jpg", ".json") in background_annotation_file_list]
    
    try:
        # ========== 对于每一张背景图片，开始贴图操作： ==========
        for bg_img_name in tqdm(background_list):

            # ========== 处理LabelME文件中已有的标注： ==========
            if copy_existing_annotations: # TODO 10.29 支持拷贝背景图像中原有的标注
                annotation_filename = bg_img_name.replace(".jpg", ".json")
                data_annotation_filepath = os.path.join(background_annotations_folder, annotation_filename)
                existing_label_list_labelme_format = load_data_annotation(data_annotation_filepath)
                existing_label_list = []
                if existing_label_list_labelme_format:
                    for existing_obj in existing_label_list_labelme_format:
                        _cls_name = existing_obj["label"]
                        if not _cls_name in class_name_id_mapping:
                            class_name_id_mapping[_cls_name] = len(class_name_id_mapping)
                        _shape_type = existing_obj["shape_type"]
                        if _shape_type == "polygon":
                            existing_label_list.append({
                                "class": _cls_name,
                                "class_id": class_name_id_mapping[_cls_name],
                                "polygon": existing_obj["points"],
                                "bbox": polygon_to_bbox(existing_obj["points"]),
                            })
                        elif _shape_type == "rectangle":
                            existing_label_list.append({
                                "class": _cls_name,
                                "class_id": class_name_id_mapping[_cls_name],
                                "polygon": bbox_to_4_points_polygon(existing_obj["points"]),
                                "bbox": polygon_to_bbox(bbox_to_4_points_polygon(existing_obj["points"])),
                            })

            for count in range(output_num_per_img):  # TODO 10.29 支持一张背景采样生成多张处理后图片

                bg_img_path = os.path.join(background_img_folder, bg_img_name)
                BG = Image.open(bg_img_path).convert("RGBA")
                bg_w, bg_h = BG.size

                bg_size_maximum = max(bg_w, bg_h)
                scaling_factor = bg_size_maximum / SCALING_CONSTANT
                # print(bg_w, bg_h, scaling_factor, )

                all_annotations: list[dict] = []
                if copy_existing_annotations:
                    all_annotations += existing_label_list

                # ========== 对于实验设定中写定的每一类物体： ==========
                for cls_id, (cls_name, png_or_folder_list) in enumerate(object_dict.items()):
                    # ========== 支持读取一个或多个folder中的PNG图片 ==========
                    png_list = []
                    for png_or_folder_path in png_or_folder_list:
                        if not os.path.exists(png_or_folder_path):
                            # continue
                            if os.path.exists(os.path.join(work_dir, png_or_folder_path)):
                                png_or_folder_path = os.path.join(work_dir, png_or_folder_path)
                            if os.path.exists(os.path.join(object_img_folder, png_or_folder_path)):
                                # png_list.append(os.path.join(object_img_folder, png_or_folder_path))
                                png_or_folder_path = os.path.join(object_img_folder, png_or_folder_path)
                        if os.path.isfile(png_or_folder_path):
                            if png_or_folder_path.endswith(".png"):
                                png_list.append(png_or_folder_path)
                        elif os.path.isdir(png_or_folder_path):
                            png_filenames = [fn for fn in os.listdir(png_or_folder_path) if fn.endswith(".png")]
                            for png_filename in png_filenames:
                                png_list.append(os.path.join(png_or_folder_path, png_filename))

                    # ========== 根据各类物体对应的数量规则，随机确定每张图片贴入的物体数量，并随机选择相应数量的物体图片 ==========
                    # x = random.randint(0, 3)  # 每类最多贴 3 个
                    number_of_obj = number_rules.get(cls_name, [DEFAULT_MINIMUM_OBJ_NUM, DEFAULT_MAXIMUM_OBJ_NUM])
                    minimum_obj_per_img = number_of_obj[0]
                    maximum_obj_per_img = number_of_obj[1]
                    x = random.randint(minimum_obj_per_img, maximum_obj_per_img)
                    chosen_pngs = random.sample(png_list, min(x, len(png_list)))

                    # ========== 对于选中的每一张物体（前景）图片 ==========
                    for obj_img_name in chosen_pngs:
                        # obj_img_path = os.path.join(object_img_folder, obj_img_name)
                        obj_img_path = obj_img_name  # 9.26
                        # TODO
                        if not os.path.exists(obj_img_path):
                            continue
                        
                        # ========== 按照透明度通道打开前景图片 ==========
                        obj_img = Image.open(obj_img_path).convert("RGBA")

                        original_obj_w, original_obj_h = obj_img.size

                        if data_creation_rules:  # TODO 根据规则约束调整贴图位置和大小
                            annotation_filename = bg_img_name.replace(".jpg", ".json")
                            data_annotation_filepath = os.path.join(background_annotations_folder, annotation_filename)
                            # if os.path.exists(data_annotation_filepath):  # TODO 应允许不带标注文件也可以执行规则
                            # position_rule, size_rule, angle_rule, add_this_obj = 
                            copy_paste_params = choose_position_scale_angle_depend_on_rule(
                                cls_name, 
                                data_creation_rules, 
                                data_annotation_filepath, 
                                obj_img.width, 
                                obj_img.height,
                                bg_w,
                                bg_h
                            )
                            position_rule = copy_paste_params["position"]
                            size_rule = copy_paste_params["size"]
                            angle_rule = copy_paste_params["angle"]
                            add_this_obj = copy_paste_params["add_this_obj"]
                            if not add_this_obj:
                                continue
                        else:
                            position_rule, size_rule, angle_rule = None, None, None

                        print(f"""当前物体使用的贴图条件：
                            类别：{cls_name}
                            文件名：{obj_img_path}
                            位置：{position_rule}
                            尺寸：{size_rule}
                            旋转角：{angle_rule}
                            """
                        )

                        # ========== 获取 alpha mask 并提取原始轮廓 ==========
                        ### PIL.Image.Image.split(): 分离出多通道图像的每个通道的信息，例如RGBA模式的最后一个通道是A通道，通过这种方式提取出物体（前景）图像的透明度通道（A通道）
                        alpha = np.array(obj_img.split()[-1])
                        ### 代码说明：
                        """
                        灰度图指的是每个像素只有一个亮度值，范围通常是 0~255。
                        有些图像的 alpha 通道确实是「0 表示完全透明，255 表示完全不透明」，中间值不用，那它就天然是一个二值掩膜。
                        实际情况：很多图像的 alpha 通道是 半透明的灰度图. 这时候 alpha 就不是二值的，而是连续的灰度值。
                        - src：输入的灰度图像。
                        - thresh：阈值。
                        - maxval：当像素值满足条件时赋予的值。
                        - type：阈值化的类型（这里是 cv2.THRESH_BINARY）。
                        -> 返回两个值：retval（实际使用的阈值），dst（二值化后的图像）。
                        """
                        _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

                        ### 代码说明：
                        """
                        这段代码用于 寻找轮廓（contours），通常在图像处理和计算机视觉中，用于从二值图像中提取出物体的边界。
                        - binary：
                        这是输入图像，应该是一个二值图像（黑白图像）。在你的代码中，这个图像可能是通过 cv2.threshold() 生成的二值图像，其中黑色（0）表示背景，白色（255）表示前景。
                        cv2.findContours() 主要是用来从这种二值图像中提取前景物体的边缘轮廓。

                        - cv2.RETR_EXTERNAL：这是轮廓检索模式，表示 只检索外部轮廓（即图像中物体的外部边界）。
                        还有其他检索模式，如 cv2.RETR_TREE（检索所有层次的轮廓），cv2.RETR_LIST（只检索顶层轮廓）。

                        - cv2.CHAIN_APPROX_SIMPLE：这是轮廓逼近方法。cv2.CHAIN_APPROX_SIMPLE 会对轮廓进行 简化，去掉轮廓中的冗余点，保留轮廓的角点。
                        如果使用 cv2.CHAIN_APPROX_NONE，则每个轮廓的每个像素点都会被保留，形成更精确的轮廓，但数据量会更大。

                        -> 返回值：
                        cv2.findContours 返回两个值，第一个是轮廓，第二个是层级（hierarchy），通常可以忽略（用 _ 占位）。
                        - contours：
                        这是一个 列表，其中每个元素表示图像中的一个轮廓。每个轮廓是一个 点集（一个包含轮廓点坐标的数组），可以用来描述物体的边界。
                        """
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if not contours:  # ??? why?
                            continue
                        
                        # 1. 获取原始轮廓（在物体图像局部坐标系中）
                        ### 代码说明：
                        """
                        contours：是 cv2.findContours() 返回的轮廓列表，里面可能有多个轮廓。

                        max(..., key=cv2.contourArea)：
                        Python 内置的 max() 可以接收一个 key 参数，用来指定“比较大小的标准”。
                        这里的标准是 cv2.contourArea，也就是轮廓的面积。
                        这行代码的作用就是：
                        在所有轮廓里，找到面积最大的那个轮廓，并赋值给 contour。

                        points = contour.squeeze()
                        contour 本质上是一个 numpy 数组，形状通常是 (N, 1, 2)：
                        N：轮廓上的点数。
                        1：OpenCV 存储时多加的一维。
                        2：每个点的 (x, y) 坐标。
                        例如：
                        array([[[10, 20]],
                               [[30, 20]],
                               [[30, 40]],
                               [[10, 40]]], dtype=int32)
                        .squeeze()：去掉 长度为 1 的维度，把 (N, 1, 2) 压缩成 (N, 2)。
                        """
                        contour: np.ndarray = max(contours, key=cv2.contourArea)  # 取最大轮廓
                        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)  # 得到BoundingBox
                        contour_points: np.ndarray = contour.squeeze()

                        # 2. 原图像中心（缩放和旋转都是围绕这个中心进行）
                        img_center = (obj_img.width / 2, obj_img.height / 2)

                        # 随机变换 
                        # 3. 随机变换参数
                        # TODO 根据约束规则生成三个贴图参数
                        if size_rule:
                            # scale = min([size_rule[0] / obj_img.width, size_rule[1] / obj_img.height])
                            scale = size_rule
                        else:
                            scale = random.uniform(0.9, 1.2)
                            scale = scale * scaling_factor
                        # scale = 1
                        # TODO 不论是否有size_rule，必须保证粘贴物体的大小不能大于背景图；否则会在后续确定center_x, center_y时报错 9.24

                        # print(obj_img.width * scale)
                        # print(obj_img.width)
                        # print(scale)
                        # print(bg_w)
                        # print(obj_img.height * scale)
                        # print(obj_img.height)
                        # print(scale)
                        # print(bg_h)
                        if obj_img.width * scale > bg_w or obj_img.height * scale > bg_h:
                            scale = min(bg_w / obj_img.width, bg_h / obj_img.height)

                        if angle_rule != None:  # Warning
                            angle = angle_rule
                        else:
                            angle = random.uniform(0, 360)
                        # angle = 0

                        if position_rule:
                            center_x = int(position_rule[0])
                            center_y = int(position_rule[1])
                        else:
                            center_x = random.randint(int(1/2 * obj_img.width * scale), int(bg_w - 1/2 * obj_img.width * scale))
                            center_y = random.randint(int(1/2 * obj_img.height * scale), int(bg_h - 1/2 * obj_img.height * scale))
                        # center_x = 0
                        # center_y = 0

                        # 4. PIL 对图片执行缩放和旋转变换
                        ### 代码说明：
                        """参数含义：
                        - angle：旋转角度（单位：度）。
                        角度是 逆时针 方向的，比如 90 就是逆时针旋转 90°。
                        如果你要顺时针，可以传负数，例如 -90。
                        - expand=True
                        控制旋转后的图像大小：
                        expand=False（默认）：保持原图大小，超出边界的部分会被裁掉。
                        expand=True：自动扩大画布，使得旋转后的图像完整保留，不会被裁剪。
                        旋转并自动扩展画布，保证图像完整显示。
                        """
                        obj_img = obj_img.resize((int(obj_img.width * scale), int(obj_img.height * scale)))
                        obj_w_after_scale_before_rotate, obj_h_after_scale_before_rotate = obj_img.size
                        obj_img = obj_img.rotate(angle, expand=True)

                        # 5. 生成仿射变换矩阵
                        M = get_affine_matrix(center=img_center, scale=scale, angle_deg=angle)

                        # 6. 对轮廓点执行相同仿射变换
                        transformed_contour_points: np.ndarray = apply_affine_transform(contour_points, M)

                        # 8. 对轮廓点加上 paste 偏移（从物体局部坐标映射到背景图全图坐标）
                        # scale_offset_x, scale_offset_y = get_scaling_offset(int(obj_img.width), int(obj_img.height), scale)
                        scale_offset_x, scale_offset_y = get_scaling_offset(original_obj_w, original_obj_h, scale)
                        # rotation_offset_x, rotation_offset_y = get_rotation_offset(int(obj_img.width), int(obj_img.height), angle)  
                        # Notice! 旋转后本身的obj_img.width和obj_img.height也都会改变。实际上这里应该输入的宽、高是缩放后、旋转前的值。既不应该是旋转后的值，也不应该是原始图片的值
                        rotation_offset_x, rotation_offset_y = get_rotation_offset(obj_w_after_scale_before_rotate, obj_h_after_scale_before_rotate, angle)

                        transformed_contour_points: np.ndarray = transformed_contour_points + np.array([center_x, center_y])
                        transformed_contour_points_rectified_scale: np.ndarray = transformed_contour_points + np.array([scale_offset_x, scale_offset_y])
                        transformed_contour_points_rectified_rotate: np.ndarray = transformed_contour_points_rectified_scale + np.array([rotation_offset_x, rotation_offset_y])
                        # new_points = transformed_points + np.array([paste_x + 0.5 * obj_w * scale, paste_y +  0.5 * obj_h * scale])
                        # new_points = new_points - np.array([int(1/2 * obj_img.width), int(1/2 * obj_img.height)])
                        transformed_contour_points_final = transformed_contour_points_rectified_rotate

                        # ========== 执行贴图（alpha合成） ==========
                        # ========== TODO 9.24 将前景PNG粘贴到背景图上，加入羽化 + 边缘模糊 + 亮度调整 ==========
                        if do_feather or do_edge_blur or do_brightness_scale:
                            bg_numpy = do_blur_and_scale(
                                background_path_or_img = BG, 
                                foreground_path_or_img = obj_img,  
                                output_path = None,
                                pos = (center_x, center_y), 
                                do_feather = do_feather,
                                do_edge_blur = do_edge_blur,
                                do_brightness_scale = do_brightness_scale,
                                feather_radius = feather_radius, # 必须是奇数 
                                brightness_factor = 1.1,
                                do_match_histograms = True,
                            )

                            BG = numpy_to_pil(bg_numpy)
                            BG = BG.convert("RGBA")
                        else:
                            BG.alpha_composite(obj_img, dest=(center_x, center_y))

                        # ========== TODO 9.22 多边形与矩形裁剪 空 → 物体完全在外面，丢弃；多个多边形（MultiPolygon） → 丢弃 ==========
                        transformed_contour_points_final = clip_polygon_to_image(transformed_contour_points_final, bg_w, bg_h)

                        # ========== 保存标注 ==========
                        if transformed_contour_points_final is not None:  # TODO transformed_contour_points_final有可能是None，需要做此判断
                            all_annotations.append({
                                "class": cls_name,
                                "class_id": cls_id,
                                "polygon": transformed_contour_points_final.tolist(),
                                "bbox": polygon_to_bbox(transformed_contour_points_final.tolist()),
                            })

                # ========== 保存最终图像和LabelMe格式、YOLO格式标注 ==========
                out_img_name = f"synthetic_{count}_{bg_img_name}"  # TODO 10.29 支持一张背景采样生成多张处理后图片
                BG.convert("RGB").save(os.path.join(output_img_dir, out_img_name))

                # ========== JSON格式 ==========
                output_jsonfilename = os.path.join(output_ann_dir, f"{Path(out_img_name).stem}.json")
                with open(output_jsonfilename, "w", encoding="utf-8") as f:
                    json.dump(all_annotations, f, indent=2)

                # ========== YOLO bbox格式 ==========
                output_yolofilename = os.path.join(output_yolo_bbox_ann_dir, f"{Path(out_img_name).stem}.txt")
                with open(output_yolofilename, "w", encoding="utf-8") as f:
                    for obj in all_annotations:
                        f.write(" ".join([str(i) for i in bbox_to_yolo(obj["bbox"], bg_w, bg_h, class_id=obj["class_id"])]))
                        f.write("\n")

                # ========== YOLO segmentation格式 ==========
                output_yolo_seg_filename = os.path.join(output_yolo_seg_ann_dir, f"{Path(out_img_name).stem}.txt")
                with open(output_yolo_seg_filename, "w", encoding="utf-8") as f:
                    for obj in all_annotations:
                        f.write(" ".join([str(i) for i in bbox_to_yolo_segmentation(obj["polygon"], bg_w, bg_h, class_id=obj["class_id"])]))
                        f.write("\n")
    except Exception as e:
        print(f"由于发生以下错误，程序运行中途停止：{e}")
    
    finally:
        # ========== 生成类名映射表 ==========
        output_classnames_filename = os.path.join(output_folder_dir, "classes.txt")
        class_id_name_mapping = {v:k for k, v in class_name_id_mapping.items()}
        class_id_name_mapping = dict(sorted(class_id_name_mapping.items(), key=lambda x: x[0]))
        with open(output_classnames_filename, "w", encoding="utf-8") as f:
            for cls_id, cls_name in class_id_name_mapping.items():
                f.write(f"{cls_id}\t{cls_name}\n")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频抽帧保存为图片")
    parser.add_argument("--work_dir", type=str, required=True, help="工作目录")
    parser.add_argument("--output_folder_name", type=str, default="output", help="输出文件夹，如果不传入，则默认在输入文件夹中新建")
    parser.add_argument("--config_filename", type=str, default="data.yaml", help="")
    parser.add_argument("--do_feather", action="store_true", help="是否使用羽化. store_true → 默认 False，出现时变 True")
    parser.add_argument("--do_edge_blur", action="store_true", help="是否使用边缘模糊. store_true → 默认 False，出现时变 True")
    parser.add_argument("--do_brightness_scale", action="store_true", help="是否使用亮度调整. store_true → 默认 False，出现时变 True")
    parser.add_argument("--feather_radius", type=int, default=5, help="羽化半径必须是奇数")
    
    parser.add_argument("--output_num_per_img", type=int, default=1, help="每张背景图片产生几张贴图后的输出图片")
    parser.add_argument("--copy_existing_annotations", action="store_true", help="是否拷贝原有的LabelME标签（如果有的话）. store_true → 默认 False，出现时变 True")
    args = parser.parse_args()
    
    main(
        work_dir = args.work_dir, 
        output_folder_name = args.output_folder_name, 
        config_filename = args.config_filename, 
        # annotation_mode: str = "segmentation",
        do_feather = args.do_feather,
        do_edge_blur = args.do_edge_blur,
        do_brightness_scale = args.do_brightness_scale,
        feather_radius = args.feather_radius,
        output_num_per_img = args.output_num_per_img,
        copy_existing_annotations = args.copy_existing_annotations,
    )
    