import cv2
import numpy as np
from PIL import Image

def get_contour_from_png(png_path):
    # 读取带Alpha通道的图像
    img = Image.open(png_path).convert("RGBA")
    alpha = np.array(img.split()[-1])  # 获取Alpha通道

    # 将Alpha转为二值图像
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓（contours）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours_on_png(png_path, save_path=None):
    # 用 PIL 打开 PNG 图像（保持 Alpha 通道）
    img_rgba = Image.open(png_path).convert("RGBA")
    img_np = np.array(img_rgba)

    # 拆分 RGBA 通道
    bgr = cv2.cvtColor(img_np[..., :3], cv2.COLOR_RGB2BGR)
    alpha = img_np[..., 3]

    # 创建二值图（用于找轮廓）
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在 BGR 图像上画出轮廓
    result = bgr.copy()
    cv2.drawContours(result, contours, -1, color=(0, 255, 0), thickness=2)

    # 显示图像（可选）
    # cv2.imshow("Contours", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存图像（可选）
    if save_path:
        cv2.imwrite(save_path, result)
        


def draw_contours_on_png(png_path, save_path):
    # 打开原始 PNG（含透明通道）
    img_rgba = Image.open(png_path).convert("RGBA")
    img_np = np.array(img_rgba)

    # 提取通道
    rgb = img_np[..., :3]
    alpha = img_np[..., 3]

    # 创建用于绘图的背景（带 Alpha）
    result_rgba = np.dstack([rgb, alpha]).copy()

    # 找轮廓（基于 Alpha）
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建临时 BGR 图来画轮廓
    bgr_tmp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr_tmp, contours, -1, (0, 0, 255), 2)  # 红色轮廓

    # 转回 RGB，叠加原来的 Alpha 通道
    result_rgb = cv2.cvtColor(bgr_tmp, cv2.COLOR_BGR2RGB)
    final_rgba = np.dstack([result_rgb, alpha])

    # 保存为带透明通道的 PNG
    result_img = Image.fromarray(final_rgba, mode="RGBA")
    result_img.save(save_path)
    
    
if __name__ == "__main__":
    # 示例用法
    draw_contours_on_png("../example/object_img/plasticbag_1.png", save_path="../example/object_with_contours_plasticbag_1.png")
    