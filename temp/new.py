def main(
    work_dir, 
    output_folder_name="output", 
    config_filename="data.yaml", 
    annotation_mode="segmentation"
):
    
    assert annotation_mode in ["detection", "segmentation"]
    
    data_config_filename = os.path.join(work_dir, config_filename)
    output_img_dir = os.path.join(work_dir, output_folder_name, "images")
    output_ann_dir = os.path.join(work_dir, output_folder_name, "annotations")
    output_yolo_ann_dir = os.path.join(work_dir, output_folder_name, "labels")
    background_img_folder = os.path.join(work_dir, "background_img")
    # TODO
    background_annotations_folder = os.path.join(work_dir, "background_annotations")
    
    object_img_folder = os.path.join(work_dir, "object_img")
    
    with open(data_config_filename, "r") as f:
        object_dict = yaml.safe_load(f)
        print(f"已加载以下的物体类别贴图数据集：\n{object_dict}")
    
    # TODO
    data_creation_rules = None
    if "rules" in object_dict:
        data_creation_rules = object_dict.pop("rules")
        print("data_creation_rules", data_creation_rules)
    
    number_rules = {}
    if "number" in object_dict:
        number_rules = object_dict.pop("number")
        print("number_rules", number_rules)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)
    os.makedirs(output_yolo_ann_dir, exist_ok=True)
    
    background_list = os.listdir(background_img_folder)
    print(f"现在对背景图片共{len(background_list)}张进行贴图数据合成")
    
    if os.path.exists(background_annotations_folder):
        background_annotation_file_list = os.listdir(background_annotations_folder)
        print(f"其中，{len(background_annotation_file_list)}张图片带有匹配的标注数据")
        
        # TODO
        background_list = [fn for fn in background_list if fn.replace(".jpg", ".json") in background_annotation_file_list]
    
    
    for bg_name in tqdm(background_list):
        bg = Image.open(os.path.join(background_img_folder, bg_name)).convert("RGBA")
        bg_w, bg_h = bg.size
        
        bg_size_maximum = max(bg_w, bg_h)
        scaling_factor = bg_size_maximum / 1700
        print(bg_w, bg_h, scaling_factor, )

        all_annotations = []

        for cls_id, (cls_name, png_list) in enumerate(object_dict.items()):
            # x = random.randint(0, 2)  # 每类最多贴 3 个
            number_of_obj = number_rules.get(cls_name, [0, 2])
            x = random.randint(number_of_obj[0], number_of_obj[1])  # 每类最多贴 3 个
            
            chosen_pngs = random.sample(png_list, min(x, len(png_list)))

            for obj_name in chosen_pngs:
                obj_path = os.path.join(object_img_folder, obj_name)
                
                # TODO
                if not os.path.exists(obj_path):
                    continue
                
                    
                obj_img = Image.open(obj_path).convert("RGBA")
                
                original_obj_w, original_obj_h = obj_img.size
                
                if data_creation_rules:  # TODO 根据规则约束调整贴图位置和大小
                    annotation_filename = bg_name.replace(".jpg", ".json")
                    data_annotation_filename = os.path.join(background_annotations_folder, annotation_filename)
                    if os.path.exists(data_annotation_filename):
                        position_rule, size_rule, angle_rule, add_this_obj = choose_position_scale_angle_depend_on_rule(cls_name, data_creation_rules, data_annotation_filename, obj_img.width, obj_img.height)
                        if not add_this_obj:
                            continue
                else:
                    position_rule, size_rule, angle_rule = None, None, None
                
                print(cls_name, position_rule, size_rule, angle_rule)

                # 获取 alpha mask 并提取原始轮廓
                alpha = np.array(obj_img.split()[-1])
                _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                # 1. 获取原始轮廓（在物体图像局部坐标系中）
                contour = max(contours, key=cv2.contourArea)  # 取最大轮廓
                points = contour.squeeze()
                
                # 2. 原图像中心（缩放和旋转都是围绕这个中心进行）
                img_center = (obj_img.width / 2, obj_img.height / 2)

                # 随机变换 # 3. 随机变换参数
                # TODO 根据约束规则生成三个贴图参数
                if size_rule:
                    # scale = min([size_rule[0] / obj_img.width, size_rule[1] / obj_img.height])
                    scale = size_rule
                else:
                    scale = random.uniform(0.9, 1.2)
                    scale = scale * scaling_factor
                # scale = 1
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
                
                # # 4. PIL 图像变换
                obj_img = obj_img.resize((int(obj_img.width * scale), int(obj_img.height * scale)))
                obj_w_after_scale_before_rotate, obj_h_after_scale_before_rotate = obj_img.size
                obj_img = obj_img.rotate(angle, expand=True)
                
                # 4. 获取变换后图像尺寸
                w1, h1 = obj_img.width, obj_img.height
                center1 = (w1 / 2, h1 / 2)

                # 5. 生成仿射变换矩阵
                M = get_affine_matrix(center=img_center, scale=scale, angle_deg=angle)

                # 6. 对轮廓点执行相同仿射变换
                transformed_points = apply_affine_transform(points, M)

                # 对图片变换
                # obj_img = obj_img.resize((int(obj_img.width * scale), int(obj_img.height * scale)))
                # obj_img = obj_img.rotate(angle, expand=True)

                # 更新轮廓点
                def transform_points(points, scale, angle_deg, center):
                    angle = np.radians(angle_deg)
                    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle),  np.cos(angle)]])
                    scale_offset_x, scale_offset_y = get_scaling_offset(original_obj_w, original_obj_h, scale)
                    rotation_offset_x, rotation_offset_y = get_rotation_offset(int(obj_img.width), int(obj_img.height), angle_deg)
                    points = points * scale
                    # points = points + np.array([scale_offset_x, scale_offset_y])
                    points = (points - np.array([int(obj_img.width), int(obj_img.height)])) @ rot_mat.T
                    points = points + np.array([int(obj_img.width), int(obj_img.height)])
                    points = points + np.array([rotation_offset_x, rotation_offset_y])
                    points = points + center
                    return points

                # new_points = transform_points(points, scale, angle, (center_x, center_y))

                # 粘贴图片
                # paste_x = int(center_x - obj_img.width / 2)
                # paste_y = int(center_y - obj_img.height / 2)
                # 5. 计算 paste 偏移位置
                # 此时 obj_img 的坐标 (0,0) 在贴到 bg 时要放在：
                # paste_x = int(center_x + center1[0] - 0.5 * obj_w * scale)
                # paste_y = int(center_y + center1[1] - 0.5 * obj_h * scale)
                # paste_x = int(center_x - 0.5 * obj_w * scale)
                # paste_y = int(center_y - 0.5 * obj_h * scale)
                
                # 8. 对轮廓点加上 paste 偏移（从物体局部坐标映射到背景图全图坐标）
                # scale_offset_x, scale_offset_y = get_scaling_offset(int(obj_img.width), int(obj_img.height), scale)
                scale_offset_x, scale_offset_y = get_scaling_offset(original_obj_w, original_obj_h, scale)
                # rotation_offset_x, rotation_offset_y = get_rotation_offset(int(obj_img.width), int(obj_img.height), angle)  
                # 犯傻了，旋转后本身的obj_img.width和obj_img.height也都会改变。实际上这里应该输入的宽、高是缩放后、旋转前的值。既不应该是旋转后的值，也不应该是原始图片的值
                rotation_offset_x, rotation_offset_y = get_rotation_offset(obj_w_after_scale_before_rotate, obj_h_after_scale_before_rotate, angle)
                
                new_points = transformed_points + np.array([center_x, center_y])
                new_points = new_points + np.array([scale_offset_x, scale_offset_y])
                new_points = new_points + np.array([rotation_offset_x, rotation_offset_y])
                # new_points = transformed_points + np.array([paste_x + 0.5 * obj_w * scale, paste_y +  0.5 * obj_h * scale])
                # new_points = new_points - np.array([int(1/2 * obj_img.width), int(1/2 * obj_img.height)])
                
                # bg.alpha_composite(obj_img, dest=(paste_x, paste_y))
                bg.alpha_composite(obj_img, dest=(center_x, center_y))

                # 保存标注
                all_annotations.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "polygon": new_points.tolist(),
                    "bbox": polygon_to_bbox(new_points.tolist()),
                })

        # 保存最终图像和标注
        out_name = f"synthetic_{bg_name}"
        bg.convert("RGB").save(os.path.join(output_img_dir, out_name))
        
        output_jsonfilename = os.path.join(output_ann_dir, f"{Path(out_name).stem}.json")
        # with open(os.path.join(output_ann_dir, out_name.replace('.jpg', '.json')), "w") as f:
        with open(output_jsonfilename, "w", encoding="utf-8") as f:
            json.dump(all_annotations, f, indent=2)
        
        output_yolofilename = os.path.join(output_yolo_ann_dir, f"{Path(out_name).stem}.txt")
        with open(output_yolofilename, "w", encoding="utf-8") as f:
            for obj in all_annotations:
                f.write(" ".join([str(i) for i in bbox_to_yolo(obj["bbox"], bg_w, bg_h, class_id=obj["class_id"])]))
                f.write("\n")