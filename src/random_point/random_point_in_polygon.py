# 9/29

import random
from shapely.geometry import Point, Polygon


def random_point_in_polygon(poly_coords: list[tuple[float, float]]) -> tuple[float, float]:
    polygon = Polygon(poly_coords)
    minx, miny, maxx, maxy = polygon.bounds
    retry_counter = 0
    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        
        if polygon.contains(p):
            return (x, y)
        retry_counter += 1
        if retry_counter > 1000:
            print("经过多次尝试也没有在多边形内部取到点，请检查")
            break
    return poly_coords[0]


def random_point_not_in_polygon(poly_coords: list[tuple[float, float]]) -> tuple[float, float]:
    # TODO 10.28
    polygon = Polygon(poly_coords)
    minx, miny, maxx, maxy = polygon.bounds
    retry_counter = 0
    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        
        if not polygon.contains(p):
            return (x, y)
        retry_counter += 1
        if retry_counter > 1000:
            print("经过多次尝试也没有在多边形外部取到点，请检查")
            break
    return poly_coords[0]


if __name__ == "__main__":
    # 示例多边形
    poly_coords = [(0,0), (4,0), (4,4), (2,6), (0,4)]
    # 随机点
    pt = random_point_in_polygon(poly_coords)
    print(pt)
