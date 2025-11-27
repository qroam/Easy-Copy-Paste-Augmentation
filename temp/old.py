def main(work_dir):
    data_config_filename = os.path.join(work_dir, "data.yaml")
    output_img_dir = os.path.join(work_dir, "output", "images")
    output_ann_dir = os.path.join(work_dir, "output", "annotations")
    background_img_folder = os.path.join(work_dir, "background_img")
    object_img_folder = os.path.join(work_dir, "object_img")
    
    with open(data_config_filename, "r") as f:
        object_dict = yaml.safe_load(f)
        print(f"已加载以下的物体类别贴图数据集：\n{object_dict}")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)

    background_list = os.listdir(background_img_folder)
    print(f"现在对背景图片共{len(background_list)}张进行贴图数据合成")
    for bg_name in tqdm(background_list):
        bg = Image.open(os.path.join(background_img_folder, bg_name)).convert("RGBA")
        bg_w, bg_h = bg.size
        
        bg_size_maximum = max(bg_w, bg_h)
        scaling_factor = bg_size_maximum / 1700
        print(bg_w, bg_h, scaling_factor, )

        all_annotations = []

        for cls_id, (cls_name, png_list) in enumerate(object_dict.items()):
            x = random.randint(0, 2)  # 每类最多贴 3 个
            chosen_pngs = random.sample(png_list, min(x, len(png_list)))

            for obj_name in chosen_pngs:
                obj_path = os.path.join(object_img_folder, obj_name)
                obj_img = Image.open(obj_path).convert("RGBA")
                
                original_obj_w, original_obj_h = obj_img.size

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
                scale = random.uniform(0.9, 1.2)
                scale = scale * scaling_factor
                # scale = 1
                angle = random.uniform(0, 360)
                # angle = 0
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
                    "polygon": new_points.tolist()
                })

        # 保存最终图像和标注
        out_name = f"synthetic_{bg_name}"
        bg.convert("RGB").save(os.path.join(output_img_dir, out_name))
        
        output_jsonfilename = os.path.join(output_ann_dir, f"{Path(out_name).stem}.json")
        # with open(os.path.join(output_ann_dir, out_name.replace('.jpg', '.json')), "w") as f:
        with open(output_jsonfilename, "w", encoding="utf-8") as f:
            json.dump(all_annotations, f, indent=2)