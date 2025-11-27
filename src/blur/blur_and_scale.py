# 9.24

import cv2
import numpy as np
from PIL import Image
# from skimage.exposure import match_histograms

from typing import Union, Optional

from .utils import pil_to_numpy, numpy_to_pil

def read_or_convert_img(input_img: Union[str, np.ndarray, Image.Image], with_alpha_channel: bool = True) -> np.ndarray:
    """
    cv2.IMREAD_COLOR (默认值)
    以 彩色 方式读图。
    忽略透明通道（alpha）。
    读入结果是 3 通道 BGR。
    即使原图是灰度或带透明度的 PNG，结果都会被转成 3 通道。

    cv2.IMREAD_UNCHANGED
    原样读入，不做任何通道丢弃。
    如果图像有 alpha 通道（比如 RGBA 的 PNG），读入结果就是 4 通道 BGRA。
    如果是灰度图，就保持单通道。
    
    Args:
        input_img (Union[str, np.ndarray, Image.Image]): _description_

    Returns:
        np.ndarray: _description_
    """
    if type(input_img) == str:
        if with_alpha_channel:
            img_numpy= cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
        else:
            img_numpy= cv2.imread(input_img, cv2.IMREAD_COLOR)
    elif isinstance(input_img, Image.Image):
        if with_alpha_channel:
            img_numpy = pil_to_numpy(input_img, code=cv2.COLOR_RGB2BGRA)
            assert img_numpy.shape[-1] == 4, img_numpy.shape
        else:
            img_numpy = pil_to_numpy(input_img)
            if img_numpy.shape[-1] == 4:
                img_numpy = img_numpy[..., :3]
    else:
        assert isinstance(input_img, np.ndarray)
        img_numpy = input_img
    return img_numpy


def do_blur_and_scale(
    background_path_or_img: Union[str, np.ndarray, Image.Image], 
    foreground_path_or_img: Union[str, np.ndarray, Image.Image],  
    output_path: Optional[str] = None,
    pos: tuple[int] = (50, 50), 
    do_feather: bool = True,
    do_edge_blur: bool = True,
    do_brightness_scale: bool = True,
    feather_radius: int = 5, # 必须是奇数 
    brightness_factor: float = 1.1,
    do_match_histograms: bool = True,  # 使用直方图匹配的方式调整前景亮度
    ):
    """
    将前景PNG粘贴到背景图上，加入羽化 + 边缘模糊 + 亮度调整。
    
    参数:
    - background_path: 背景图片路径
    - foreground_path: 前景PNG路径 (带透明通道)
    - output_path: 输出结果保存路径
    - pos: 粘贴位置 (左上角)
    - feather_radius: 羽化半径。必须是奇数！！！！！！！！
    - brightness_factor: 前景亮度调整系数
    """
    # ========== 读取背景图 ==========
    bg = read_or_convert_img(background_path_or_img, with_alpha_channel=False)
    bg_h, bg_w = bg.shape[:2]

    # ========== 读取前景 (带alpha) ==========
    fg = read_or_convert_img(foreground_path_or_img, with_alpha_channel=True)
    fg_h, fg_w = fg.shape[:2]

    # ========== 拆分通道 ==========
    fg_rgb = fg[..., :3]
    fg_alpha = fg[..., 3] / 255.0  # 归一化到 [0,1]

    # ========== 亮度调整 ==========
    if do_brightness_scale:
        if do_match_histograms:
            # TODO 目前调用match_histograms报错
            # ========== 自动亮度匹配(使用直方图匹配调整前景亮度) ==========
            # x, y = pos
            # roi = bg[y:y+fg_h, x:x+fg_w]
            # if roi.shape[0] != fg_h or roi.shape[1] != fg_w:
            #     raise ValueError("前景图放置位置超出背景范围")
            # fg_rgb = match_histograms(fg_rgb, roi, channel_axis=-1).astype(np.uint8)
            fg_rgb = cv2.convertScaleAbs(fg_rgb, alpha=brightness_factor, beta=0)
        else:
            fg_rgb = cv2.convertScaleAbs(fg_rgb, alpha=brightness_factor, beta=0)

    # ========== 羽化处理 ==========
    # 对alpha通道做高斯模糊，形成羽化效果
    # print(fg_alpha.shape)
    # cv2.imwrite("../temp/fg_alpha.png", fg_alpha * 255.0)
    # with np.printoptions(threshold=np.inf):
    # #     print(fg_alpha)
    #     with open("../temp/fg_alpha.txt", "w") as f:
    #         print(fg_alpha, file=f)
    if do_feather:
        fg_alpha_feathered = cv2.GaussianBlur(fg_alpha, (feather_radius*2+1, feather_radius*2+1), 0)
    else:
        fg_alpha_feathered = fg_alpha  # TODO
    # print(fg_alpha_feathered.shape)
    # cv2.imwrite("../temp/fg_alpha_feathered.png", fg_alpha_feathered * 255.0)
    # with np.printoptions(threshold=np.inf):
    # #     print(fg_alpha_feathered)
    #     with open("../temp/fg_alpha_feathered.txt", "w") as f:
    #         print(fg_alpha_feathered, file=f)
    
    
    # ========== 边缘模糊 ==========
    # 提取边缘区域 (膨胀-腐蚀)
    if do_edge_blur:
        edge_blur_kernel = np.ones((3,3), np.uint8)
        edge = cv2.dilate(fg_alpha, edge_blur_kernel, iterations=1) - cv2.erode(fg_alpha, edge_blur_kernel, iterations=1)
        edge_blurred = cv2.GaussianBlur(edge, (5,5), 0)
    # print(edge_blurred.shape)
    # cv2.imwrite("../temp/edge_blurred.png", edge_blurred * 255.0)
    # with np.printoptions(threshold=np.inf):
    # #     print(edge_blurred)
    #     with open("../temp/edge_blurred.txt", "w") as f:
    #         print(edge_blurred, file=f)
    if do_feather and do_edge_blur: 
        fg_alpha_final = np.clip(fg_alpha_feathered + edge_blurred*0.3, 0, 1)
    elif do_edge_blur:
        fg_alpha_final = np.clip(fg_alpha_feathered + edge_blurred*0.3, 0, 1)
    else:
        fg_alpha_final = np.clip(fg_alpha_feathered, 0, 1)  # TODO 
    
    # ========== This is very important!!! ==========
    fg_alpha_binary_mask = np.where(fg_alpha < 1e-6, 0, 1)  # TODO 使用fg_alpha_binary_mask似乎没有直接使用fg_alpha的效果好
    fg_alpha_final = np.clip(fg_alpha_final * fg_alpha, 0, 1)  # TODO 让原有的遮罩中黑色的部分（0）仍保持黑色，使得物体不要污染背景
    # print(fg_alpha_final.shape)
    # cv2.imwrite("../temp/fg_alpha_final.png", fg_alpha_final * 255.0)
    # with np.printoptions(threshold=np.inf):
    # #     print(fg_alpha_final)
    #     with open("../temp/fg_alpha_final.txt", "w") as f:
    #         print(fg_alpha_final, file=f)

    # ========== 粘贴操作 ==========
    x, y = pos  # 粘贴位置 (注意是左上角)
    roi = bg[y:y+fg_h, x:x+fg_w]

    # ========== 确保ROI大小合法 ==========
    # TODO 目前这个问题经常出现
    if roi.shape[0] != fg_h or roi.shape[1] != fg_w:
        # raise ValueError("前景图放置位置超出背景范围")
        fg_rgb = fg_rgb[:roi.shape[0], :roi.shape[1], ...]  # TODO
        fg_alpha_final = fg_alpha_final[:roi.shape[0], :roi.shape[1], ...]

    # ========== alpha扩展到3通道 ==========
    fg_alpha_3c = np.dstack([fg_alpha_final]*3)

    # ========== 融合 ==========
    blended = (fg_rgb * fg_alpha_3c + roi * (1 - fg_alpha_3c)).astype(np.uint8)
    # ValueError: operands could not be broadcast together with shapes (748,273,3) (748,824,3)
    # c = input()
    
    # ========== 写回背景 ==========
    bg[y:y+fg_h, x:x+fg_w] = blended

    # ========== 保存结果 ==========
    if output_path:
        cv2.imwrite(output_path, bg)
    return bg