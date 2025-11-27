# Created in 17th, Sep
# 9.24 增添了支持以整个背景图片作为size缩放的参考对象
# TODO 9.26 增添了支持以整个背景图片作为position参考目标的功能
# TODO 10.28 增添了当标注的reference物体为segmentation格式的标注时的处理方式，同时重构了代码

import os
import json
import random
from typing import Optional

from random_point.random_point_in_polygon import random_point_in_polygon, random_point_not_in_polygon


ALL_VALID_POSITION_REF_RULES = [
    "up",
    "random",  # 如果参考对象是segmentation标注，目前只支持"random"和"out_random"
    "center",
    "out_random",  # 如果参考对象是segmentation标注，目前只支持"random"和"out_random"
]

ALL_VALID_SIZE_REF_RULES = [  # 如果参考对象是segmentation标注，目前不支持size_rule. 请将size_reference选定为整张图片"_"
    "mean",
    "largest",
    "smallest"
]

ALL_SUPPORTED_ANNOTATION_FORMATS = [
    "detection",
    "segmentation",
]


def load_data_annotation(filename, ):
    if not os.path.exists(filename):
        return None
    with open(filename, "r", encoding="utf-8") as f:
        annotation_dict = json.load(f)
    label_list = annotation_dict["shapes"]
    return label_list



# def load_reference_objects(
#     label_list: list,
#     reference_classname: str,
#     annotation_format: Optional[str] = "detection",  # TODO 10.28
# ):
#     reference_class_label_list = [dp for dp in label_list if dp["label"] == reference_classname]
    
#     if annotation_format == "segmentation":
#         pass
#     elif annotation_format == "detection":
#         pass
#     else:
#         raise ValueError(f"annotation_format {annotation_format} should be one of the followings: {ALL_SUPPORTED_ANNOTATION_FORMATS}")
    
    
#     if reference_class_label_list:
#         all_widths = []
#         all_heights = []
#         for reference in reference_class_label_list:
#             bbox_width = abs(reference["points"][0][0] - reference["points"][1][0])
#             bbox_height = abs(reference["points"][0][1] - reference["points"][1][1])
#             all_widths.append(bbox_width)
#             all_heights.append(bbox_height)
#             # TODO 10.28 需要加入当size_reference为segmentation格式的标注时的处理方式
    
#     chosen_reference = random.sample(position_reference_class_label_list, 1)[0]
#     # print(chosen_reference)
#     bbox_width = abs(chosen_reference["points"][0][0] - chosen_reference["points"][1][0])
#     bbox_height = abs(chosen_reference["points"][0][1] - chosen_reference["points"][1][1])
#     bbox_center = (
#         0.5 * (chosen_reference["points"][0][0] + chosen_reference["points"][1][0]), 
#         0.5 * (chosen_reference["points"][0][1] + chosen_reference["points"][1][1])
#     )
#     bbox_uppermost = bbox_center[1] - 0.5 * bbox_height
#     bbox_leftmost = bbox_center[0] - 0.5 * bbox_width
#     bbox_bottommost = bbox_center[1] + 0.5 * bbox_height
#     bbox_rightmost = bbox_center[0] + 0.5 * bbox_width
    
#     else:  # 当前图片不存在该类物体的标记
#         return
    


def solve_size_rule_for_detection_anno(
    params: dict,
    scale: float,
    obj_width: float,
    obj_height: float,
    size_reference_class_label_list: list,
    size_reference_rule: str,
) -> float:
    all_widths = []
    all_heights = []
    for reference in size_reference_class_label_list:
        bbox_width = abs(reference["points"][0][0] - reference["points"][1][0])
        bbox_height = abs(reference["points"][0][1] - reference["points"][1][1])
        all_widths.append(bbox_width)
        all_heights.append(bbox_height)
        # TODO 10.28 需要加入当size_reference为segmentation格式的标注时的处理方式

    if size_reference_rule == "mean":
        # print(all_widths, all_heights)
        w = sum(all_widths) / len(all_widths)
        h = sum(all_heights) / len(all_heights)
    elif size_reference_rule == "largest":
        w = max(all_widths)
        h = max(all_heights)
    elif size_reference_rule == "smallest":
        w = min(all_widths)
        h = min(all_heights)
    # size = (scale * w, scale * h)
    else:
        raise ValueError(f"size_reference_rule {size_reference_rule} should be one of the followings: {ALL_VALID_SIZE_REF_RULES}")

    size = min([scale * w / obj_width, scale * h / obj_height])
    params.update({
        "size": size,
    })
    return size


def solve_position_rule_for_detection_anno(
    params: dict,
    # scale: float,
    obj_width: float,
    obj_height: float,
    position_reference_class_label_list: list,
    position_reference_rule: str,
) -> tuple[int, int]:
    # TODO 特别要注意这里在决定position的同时对size也进行了调整！！！
    chosen_reference = random.sample(position_reference_class_label_list, 1)[0]
    # print(chosen_reference)
    bbox_width = abs(chosen_reference["points"][0][0] - chosen_reference["points"][1][0])
    bbox_height = abs(chosen_reference["points"][0][1] - chosen_reference["points"][1][1])
    bbox_center = (
        0.5 * (chosen_reference["points"][0][0] + chosen_reference["points"][1][0]), 
        0.5 * (chosen_reference["points"][0][1] + chosen_reference["points"][1][1])
    )
    bbox_uppermost = bbox_center[1] - 0.5 * bbox_height
    bbox_leftmost = bbox_center[0] - 0.5 * bbox_width
    bbox_bottommost = bbox_center[1] + 0.5 * bbox_height
    bbox_rightmost = bbox_center[0] + 0.5 * bbox_width
    
    # TODO 10.28 需要加入当position_reference为segmentation格式的标注时的处理方式

    if position_reference_rule == "up":
        size = (bbox_width, bbox_height)
        size = min([size[0] / obj_width, size[1] / obj_height])

        # position = (bbox_center[0] - 1/2 * obj_width * size, bbox_uppermost - 1/2 * obj_height * size)
        position = (bbox_leftmost, bbox_uppermost)

    elif position_reference_rule == "random":
        size = (bbox_width, bbox_height)
        size = min([size[0] / obj_width, size[1] / obj_height])
        position = (
            random.randint(int(bbox_leftmost + 1/2 * obj_width * size), int(bbox_rightmost - 1/2 * obj_width * size)) - 1/2 * obj_width * size,
            random.randint(int(bbox_uppermost + 1/2 * obj_height * size), int(bbox_bottommost - 1/2 * obj_height * size)) - 1/2 * obj_height * size,
        )

    elif position_reference_rule == "center":
        size = (bbox_width, bbox_height)
        size = min([size[0] / obj_width, size[1] / obj_height])
        position = (bbox_center[0] - 1/2 * obj_width * size, bbox_center[1] - 1/2 * obj_height * size)
    else:
        raise ValueError(f"position_reference_rule {position_reference_rule} should be one of the followings: {ALL_VALID_POSITION_REF_RULES}")
    params.update({
        "size": size,
        "position": position,
    })
    return position


def solve_position_rule_for_segmentation_anno(
    params: dict,
    # scale: float,
    obj_width: float,
    obj_height: float,
    position_reference_class_label_list: list,
    position_reference_rule: str,
) -> tuple[int, int]:
    # TODO 特别要注意这里在决定position的同时对size也进行了调整！！！
    chosen_reference = random.sample(position_reference_class_label_list, 1)[0]
    chosen_reference_points = chosen_reference["points"]
    
    # TODO 10.28 需要加入当position_reference为segmentation格式的标注时的处理方式

    if position_reference_rule == "random":
        position = random_point_in_polygon(chosen_reference_points)
    elif position_reference_rule == "out_random":
        position = random_point_not_in_polygon(chosen_reference_points)
    elif position_reference_rule == "up":
        raise ValueError(f"position_reference_rule {position_reference_rule} is not suppoted for segmentation annotation format")
    elif position_reference_rule == "center":
        raise ValueError(f"position_reference_rule {position_reference_rule} is not suppoted for segmentation annotation format")
    else:
        raise ValueError(f"position_reference_rule {position_reference_rule} should be one of the followings: {ALL_VALID_POSITION_REF_RULES}")
    params.update({
        # "size": size,
        "position": position,
    })
    return position



def choose_position_scale_angle_depend_on_rule(
    cls_name: str, 
    data_creation_rules: str, 
    data_annotation_filename: str, 
    obj_width: float, 
    obj_height: float, 
    bg_width: float, 
    bg_height: float, 
    # annotation_format: Optional[str] = "detection",  # TODO 10.28
):
    """
    10th, July
    Args:
        cls_name (_type_): 贴图的类别名，例如“气球”
        data_creation_rules (_type_): 从data.yaml加载的一部分，一个字典。用于说明不同类型的贴图应用的规则
        data_annotation_filename (_type_): 背景图片的标注文件，一个JSON文件，labelme格式标注

    Returns:
        _type_: _description_
    """
    # TODO 10th, July
    
    position: Optional[tuple[int, int]] = None
    size: Optional[float] = None
    angle: Optional[float] = None
    add_this_obj: bool = True
    
    params = {
        "position": position,
        "size": size,
        "angle": angle,
        "add_this_obj": add_this_obj,
    }
    
    label_list = load_data_annotation(data_annotation_filename)
    # if not label_list:  # 该底图没有对应的标注文件   # TODO 应允许不带标注文件也可以执行规则
    #     return None, None, None, True
    
    if not cls_name in data_creation_rules:  # 该贴图没有对应的约束规则
        # print(11111)
        # return None, None, None, True
        return params
    rule = data_creation_rules[cls_name]
    
    if "size" in rule:
        size_reference_class = rule["size"]["reference"]
        size_reference_class_anno_format = rule["size"].get("anno_format", "detection")  # TODO 应该运行同一张底图中同时有不同格式的标注，同时size_reference和position_reference可以自由选择参照不同格式的标注物体
        size_reference_rule = rule["size"].get("rule", "mean")
        size_reference_scale = rule["size"].get("scale", [0.9, 1.1])
        scale = random.uniform(size_reference_scale[0], size_reference_scale[1])
        
        # TODO 9.24 增添了支持以整个背景图片作为size缩放的参考对象
        if size_reference_class == "_":
            size: float = min([scale * bg_width / obj_width, scale * bg_height / obj_height])
            # print(size)
            params["size"] = size
        else:
            # if annotation_format == "detection":
            if size_reference_class_anno_format == "detection":
                size_reference_class_label_list = [dp for dp in label_list if dp["label"] == size_reference_class]
                # TODO
                if size_reference_class_label_list:
                    size = solve_size_rule_for_detection_anno(params, scale, obj_width, obj_height, size_reference_class_label_list, size_reference_rule,)
            # elif annotation_format == "segmentation":
            elif size_reference_class_anno_format == "segmentation":
                raise ValueError('如果参考对象是segmentation标注，目前不支持size_rule. 请将size_reference选定为整张图片"_"')
            else:
                raise ValueError(f"annotation_format {size_reference_class_anno_format} should be one of the followings: {ALL_SUPPORTED_ANNOTATION_FORMATS}")
    
    if "position" in rule:
        position_reference_class = rule["position"]["reference"]
        position_reference_class_anno_format = rule["position"].get("anno_format", "detection")  # TODO 应该运行同一张底图中同时有不同格式的标注，同时size_reference和position_reference可以自由选择参照不同格式的标注物体
        position_reference_rule = rule["position"].get("rule", "center")
        position_reference_value = rule["position"].get("value", None)
        # assert position_reference_rule == "in"
        
        # TODO 9.26 增添了支持以整个背景图片作为position参考目标的功能
        if position_reference_class == "_":
            assert position_reference_value is not None
            w1, h1, w2, h2 = position_reference_value
            if w1 < w2:
                bbox_leftmost = w1 * bg_width
                bbox_rightmost = w2 * bg_width
            else:
                bbox_leftmost = w2 * bg_width
                bbox_rightmost = w1 * bg_width
            if h1 < h2:
                bbox_uppermost = h1 * bg_height
                bbox_bottommost = h2 * bg_height
            else:
                bbox_uppermost = h2 * bg_height
                bbox_bottommost = h1 * bg_height
            position = (
                    random.randint(int(bbox_leftmost + 1/2 * obj_width * size), int(bbox_rightmost - 1/2 * obj_width * size)) - 1/2 * obj_width * size,
                    random.randint(int(bbox_uppermost + 1/2 * obj_height * size), int(bbox_bottommost - 1/2 * obj_height * size)) - 1/2 * obj_height * size,
                )
            params["position"] = position
        else:
            if position_reference_class_anno_format == "detection":
                position_reference_class_label_list = [dp for dp in label_list if dp["label"] == position_reference_class]
                if not position_reference_class_label_list:  # 对于强制贴入某物体的对象，如果该图片中不存在该物体，那就不能贴入
                    # return None, None, None, False
                    params["add_this_obj"] = False
                    return params
                position = solve_position_rule_for_detection_anno(params, obj_width, obj_height, position_reference_class_label_list, position_reference_rule)
                
            elif position_reference_class_anno_format == "segmentation":
                position_reference_class_label_list = [dp for dp in label_list if dp["label"] == position_reference_class]
                if not position_reference_class_label_list:  # 对于强制贴入某物体的对象，如果该图片中不存在该物体，那就不能贴入
                    # return None, None, None, False
                    params["add_this_obj"] = False
                    return params
                position = solve_position_rule_for_segmentation_anno(params, obj_width, obj_height, position_reference_class_label_list, position_reference_rule)
            else:
                raise ValueError(f"annotation_format {position_reference_class_anno_format} should be one of the followings: {ALL_SUPPORTED_ANNOTATION_FORMATS}")
            
            
    if "angle" in rule:  # 旋转角度规则暂不适用
        angle = rule["angle"].get("value", 0)  # 但藤蔓和鸟巢应默认0度旋转（不旋转）
        params["angle"] = angle
    
    # return position, size, angle, add_this_obj
    return params


