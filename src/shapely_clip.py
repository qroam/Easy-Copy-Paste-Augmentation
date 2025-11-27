# 22nd, Sep

# 解决思路
# 把 polygon 转换为 shapely.geometry.Polygon。
# 定义背景图的矩形边界（0,0)–(W,H)，同样转为 shapely.geometry.Polygon。
# 使用 poly.intersection(rect) 计算裁剪后的多边形。
# 如果结果是空 → 物体完全在外面，丢弃。
# 如果结果是多个多边形（MultiPolygon） → 丢弃（你要求只保留连通的物体）。
# 如果结果是一个合法 Polygon → 取这个 polygon 的点，归一化到 [0,1]，保存。
# 返回处理后的多边形坐标。

# Dependencies: On my local PC, `shapely` has been installed on (base) environment

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def clip_polygon_to_image(points, img_w: int, img_h: int) -> np.ndarray:
    """
    将 polygon 裁剪到图像边界，保证结果合法
    points: [(x1, y1), (x2, y2), ...]  原始多边形点（像素坐标）
    img_w, img_h: 背景图像的宽和高
    return: [(x, y), ...] 裁剪并归一化后的多边形点，如果不合法返回 None
    """
    # ========== 原始多边形 ==========
    # poly = Polygon(points)
    # if not poly.is_valid:
    #     poly = poly.buffer(0)  # 修复自交等问题（拓扑错误）
    try:
        poly = Polygon(points)
        poly = poly.buffer(0)  # 尝试修复拓扑错误
    except Exception as e:
        print(f"Polygon creation failed: {e}")
        return []
    
    # ========== 背景矩形边界 ==========
    rect = box(0, 0, img_w, img_h)  # 等价于 [(0,0), (0,h), (w,h), (w,0)]
    
    # ========== 取交集（裁剪） ==========
    intersection = poly.intersection(rect)
    
    # ========== 如果交集为空或不是多边形（可能是线段或点） ==========
    if intersection.is_empty:
        return None  # 完全在外面，丢弃
    
    # ========== 如果结果是多个多边形（MultiPolygon），丢弃 ==========
    if intersection.geom_type != "Polygon":
        return None
    
    # ========== 获取裁剪后的多边形坐标 ==========
    clipped_coords = np.array(intersection.exterior.coords)
    clipped_coords = np.array(intersection.exterior.coords[:-1]) # 去掉闭合点
    # 在 Shapely 里，一个 Polygon 的外边界坐标 (polygon.exterior.coords) 默认是 闭合的
    # YOLOv8 segment 的 polygon 点序列 不需要重复闭合点
        
    # 归一化到 [0,1]
    # clipped_coords[:, 0] /= img_w
    # clipped_coords[:, 1] /= img_h
    
    # return clipped_coords.tolist()
    return clipped_coords


# ========== 示例 ==========
if __name__ == "__main__":
    # 背景 640x480
    W, H = 640, 480
    
    # 一个越界的矩形 polygon
    poly_points = [(600, 100), (700, 100), (700, 300), (600, 300)]
    
    result = clip_polygon_to_image(poly_points, W, H)
    
    if result is None:
        print("丢弃该目标")
    else:
        # print("裁剪并归一化后的 polygon:")
        print("裁剪后的 polygon:")
        print(result)