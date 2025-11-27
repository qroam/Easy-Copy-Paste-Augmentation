
from shapely.geometry import Polygon, box

def clip_polygon_to_image_bounds(contour, width, height):
    """
    polygon_points: list of [x1, y1, x2, y2, ..., xn, yn]
    width, height: 图像宽高
    return: 裁剪后的多边形点 [x1, y1, x2, y2, ..., xm, ym]
    """
    # 将点对变为 (x, y) 元组
    # coords = [(polygon_points[i], polygon_points[i + 1]) for i in range(0, len(polygon_points), 2)]
       
    
    # 构造多边形和图像边界矩形
    # poly = Polygon(contour)
    try:
        poly = Polygon(contour)
        poly = poly.buffer(0)  # 尝试修复拓扑错误
    except Exception as e:
        print(f"Polygon creation failed: {e}")
        return []
    img_rect = box(0, 0, width, height)  # 等价于 [(0,0), (0,h), (w,h), (w,0)]
    
    # 取交集
    intersection = poly.intersection(img_rect)
    
    # 如果交集为空或不是多边形（可能是线段或点）
    if intersection.is_empty or not intersection.geom_type.startswith('Polygon'):
        return []

    # 如果是 MultiPolygon，选最大那个
    if intersection.geom_type == 'MultiPolygon':
        intersection = max(intersection.geoms, key=lambda g: g.area)
    
    # 获取裁剪后的点
    clipped_coords = list(intersection.exterior.coords)[:-1]  # 去掉闭合点
    # return [coord for point in clipped_coords for coord in point]
    return clipped_coords