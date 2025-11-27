# 9.24

from PIL import Image
import cv2
import numpy as np

def pil_to_numpy(img_pil: Image.Image, code = cv2.COLOR_RGB2BGR) -> np.ndarray:
    """
    PIL.Image.open() → PIL.Image.Image（RGB，类对象）
    cv2.imread() → numpy.ndarray（BGR，数组）
    OpenCV 的函数只能吃 numpy.ndarray，所以不能直接把 PIL 对象丢进去。
    解决办法：用 np.array(img_pil) 转换成数组，再 cv2.cvtColor() 转换颜色顺序。

    Args:
        img_pil (Image): _description_

    Returns:
        np.ndarray: _description_
    """
    # 读入图像（PIL）
    # img_pil = Image.open("test.png")

    # 转换为 numpy 数组
    img_np = np.array(img_pil)

    # 注意：PIL 是 RGB，而 OpenCV 是 BGR
    img_bgr = cv2.cvtColor(img_np, code=code)

    return img_bgr

def numpy_to_pil(img_numpy: np.ndarray, code = cv2.COLOR_BGR2RGB) -> Image.Image:
    img_rgb = cv2.cvtColor(img_numpy, code=code)
    img_pil_out = Image.fromarray(img_rgb)
    return img_pil_out