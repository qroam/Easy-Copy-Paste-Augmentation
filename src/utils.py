def polygon_to_bbox(polygon: list) -> list:
    """
    将多边形顶点转化为外接矩形框（bounding box）

    参数:
    polygon: list of tuples/list
        多边形的点坐标，例如 [(x1, y1), (x2, y2), ..., (xn, yn)]

    返回:
    bbox: list
        矩形框 [x_min, y_min, x_max, y_max]
    """
    if not polygon:
        raise ValueError("polygon 点集不能为空")

    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    return [x_min, y_min, x_max, y_max]


def bbox_to_4_points_polygon(bbox_labelme_format: list) -> list:
    # 10.29
    """
    将两点标注的bbox转化为四点

    参数:
    bbox_labelme_format: list of tuples/list
        bbox的点坐标，例如 [(x1, y1), (x2, y2)]

    返回:
    polygon: list
        矩形框 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    if not bbox_labelme_format:
        raise ValueError("bbox_labelme_format 点集不能为空")
    
    x1 = bbox_labelme_format[0][0]
    y1 = bbox_labelme_format[0][1]
    x2 = bbox_labelme_format[1][0]
    y2 = bbox_labelme_format[1][1]

    return [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]


def bbox_to_yolo(bbox: list, image_width: int, image_height: int, class_id: int=0) -> list:
    """
    将 bounding box 转换为 YOLO 格式

    参数:
    bbox: list 或 tuple
        [x_min, y_min, x_max, y_max] 格式的边界框
    image_width: int
        图像宽度
    image_height: int
        图像高度
    class_id: int
        类别ID，默认是0

    返回:
    yolo_bbox: list of float
        [class_id, x_center, y_center, width, height]，已归一化
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return [class_id, x_center, y_center, width, height]



def bbox_to_yolo_segmentation(polygon: list, image_width: int, image_height: int, class_id: int=0) -> list:
    """
    将 bounding box 转换为 YOLO 分割任务 (segment) 格式
    
    标准 YOLO 不支持语义分割（只有 bbox）。

    改进版 YOLO（YOLOv5-seg, YOLOv8-seg 等）支持实例分割，标注格式通常是 多边形 或 RLE mask，但不是传统的语义分割 label 图。

    如果你需要严格意义上的 语义分割（每个像素一个类别），YOLO 系列不是首选，可以考虑 U-Net/DeepLab，或者把语义分割标签转成多边形后用 YOLO 的实例分割。

    参数:
    bbox: list 或 tuple
        [x_min, y_min, x_max, y_max] 格式的边界框
    image_width: int
        图像宽度
    image_height: int
        图像高度
    class_id: int
        类别ID，默认是0

    返回:
    yolo_segmentation_bbox: list of float
        [class_id  x_center  y_center  width  height  polygon_x1 polygon_y1 polygon_x2 polygon_y2 ...]
    """
    x_min, y_min, x_max, y_max = polygon_to_bbox(polygon)
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return_list =  [class_id, x_center, y_center, width, height]
    
    for point in polygon:
        x = point[0]
        y = point[1]
        return_list += [x/image_width, y/image_height]
    
    return return_list