# 基于贴图的视觉目标检测数据合成

**English README:** [README_english.md](https://github.com/qroam/Easy-Copy-Paste-Augmentation/blob/main/docs/README_english.md)

![Image failed to load](/docs/demo.jpg "Demo")

## 快速开始
```bash
cd src/
python conditional_copy_paste.py \
    --work_dir "../example" \
    --config_filename "foreign_object_v1.yaml" \
    --output_folder_name "foreign_object_v1" \
    --do_feather \
    --do_edge_blur \
    --output_num_per_img 10 \
    --copy_existing_annotations
```

## 依赖环境
```bash
pip3 install -r requirements.txt
```

## 简介
### 背景
近年来，基于深度学习的计算机视觉目标检测技术成为了这一领域的主流方法，并在众多的工业场景中获得了落地应用。深度学习模型是数据驱动的，模型的训练依赖大量数据。如何获得足够丰富、多样的数据，以尽可能地涵盖实际应用场景可能出现的各种情况，是决定一个深度学习视觉检测项目成败的关键。模型常常受限于训练数据瓶颈，面临泛化性不足的隐患，这大大制约了工业级部署的效率。

传统的数据采集方法需要一定周期来采集大量场景相关的数据，然后开展人工标注，这是最简单粗暴的方法，耗力又耗时（labour-intensive and time-consuming）。然而事实上，需求方甚至不总是有这样的耐心，来等待算法开发者花上数个月来进行数据采集和标注。

例如，在一些制造业工厂相关的需求中，需求方希望我们开发检测各类工具器械的模型，来确保工作人员携带的工具合规。需求方可能只提供一份工具的清单，并附上一张图片，就希望我们能够开发出可供演示的模型。此时如果算法的开发者提出，“要在工厂现场布置摄像头，进行为期数个月的实地图像采集，否则办不到”，这反而听起来像是个“无理要求”了。在另一些情况下，真实样本是极难获取的，例如异常、病害或风险相关场景中，由于现实世界存在的样本多数是正常的，带来较严重的样本稀缺和类别不均衡问题。例如在铁路运输场景中，人们可能希望对铁道线上的砖石、塑料垃圾类异物进行检测，以帮助及时排除安全隐患，如果我们要采集真实样本，恐怕得找个试验场把这些东西一个一个摆上去，即使这样也收集不了太多样本。

事实上，在图片中加入一些本不存在的物体，人们生活中早有这样的需求。这就是大名鼎鼎的**PS**软件在十几年前开始火爆的原因。
而在深度学习领域，这也并不是一个新鲜的想法。早已有相关的研究论文[Ref. 1, 2]指出**基于贴图的样本合成**是一种简单易行但非常有效的数据构造方法。这样的方法在相关文献中一般有一个通用的名词：**复制粘贴样本增强**（Copy-Paste Augmentation）。

复制粘贴样本增强的好处不但在于可以快速获取一批大批量的数据，而且在于，你无需手工标注——物体的轮廓信息在贴图的过程中自动转化为标签信息。

尽管最近，生成式模型取得的最新进展（例如最近火爆出圈的Nano Banana）让这种古老的方法看起来不那么跟得上时代潮流。但考虑到最新生成式模型的*算力要求*（生成式模型通常需要一定算力资源才能实现部署，并且推理速度较慢，生成图片的吞吐量与复制粘贴样本增强相比，存在数量级差异）、*适配门槛*（例如，对于特定场景的图片生成，可能需要多次试错找到最合适的模型）、以及*生成样本和获取标注之间尚存在的距离*（可以使用生成式模型+SAM等通用推理模型搭建数据生产标注流水线，或采用可控条件生成），这些方面的制约，让复制粘贴样本增强仍然尤其一定的用武之地。

一些场景中的物体/实例识别，这些物体的识别特征是高度自包含的，物体本身作为一个整体移动后不影响其语义，因此它们天然适合通过复制粘贴样本增强构造样本。例如，异物、人员、动物。另一方面，对于高度依赖于环境信息的物体，我们推荐使用生成式模型构造样本。

### 本仓库的优势

复制粘贴样本增强虽然在工业界取得了广泛的应用，但目前易用的开源代码库并不多。它们或者较为简陋，或者被重量级库高度封装，不方便扩展。

例如，Ultralytics YOLO内置的数据增强策略虽然也提供了Copy paste方法[Ref. 5]，不过只支持在训练样本之间将已标注的实例分割遮罩互相粘贴，不能解决“从无到有”的问题，产生的图像直接用于训练，也难以进行质量控制。

为此，本仓库发布了一份复制粘贴样本增强实现，提供以下四方面的优势，来更好地控制生成合理的贴图数据：

- **贴图规则**：提供`位置（position）`、`尺寸（scale）`、`角度（angle）`三大规则控制前景物体贴入背景图片的方式
- **基于配置**：支持并推荐优先通过`yaml`**配置文件**的方式指定图片生产规则，工作目录结构清晰，灵活可控；配置文件即日志，便于后续核验
- **条件贴图**：贴图规则中，支持选择图片中已有的标注（LabelME格式）作为条件信息。例如，参考图片中已有的物体来决定贴入物体的放置位置和缩放尺寸
- **融合优化**：支持羽化、边缘模糊等处理，让前景物体更自然地融入背景，避免裁剪得到的锯齿形边缘等低层特征被模型过拟合导致在真实样本上应用性能下降

<!-- 贴入负例， -->


### 参考文献及相关代码仓库列表
1. Dwibedi et al., 2017. Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection. https://arxiv.org/pdf/1708.01642
https://github.com/debidatta/syndata-generation

2. Ghiasi et al., 2021. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. https://arxiv.org/abs/2012.07177v1

3. Zhao et al., 2023. X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion. https://proceedings.mlr.press/v202/zhao23f/zhao23f.pdf https://github.com/yoctta/XPaste

4. https://docs.ultralytics.com/guides/yolo-data-augmentation/#copy-paste-copy_paste

5. https://github.com/conradry/copy-paste-aug

6. https://github.com/AICVHub/Copy-Paste-for-Semantic-Segmentation

7. https://github.com/MarkPotanin/copy_paste_aug_detectron2

8. https://github.com/Opletts/Copy-Paste-Augmenter



## 贴图规则

贴图规则的理念是通过平移位置（x, y）、缩放比例s、旋转角r这三组参数控制前景物体贴入背景图片的方式。

1. 数量规则（number rule）：
控制每类物体贴入的数量

2. 位置约束（position rule）：
对于需要出现在特定上下文环境中的物体，设置此规则来确保物体贴在特定的范围内或特定类别的物体之上。对于可以随机出现的物体，可以不设置此规则

3. 尺寸规则（scale rule）：
控制物体贴入的缩放尺寸。可以基于整体图片设置比例，或根据图片中现存物体设置比例，确保贴入物体的大小更加符合实际

4. 角度约束（angle rule）：控制物体的旋转角度


## 配置文件
在`example/foreign_object_v1.yaml`提供了一份配置文件的实例。


## 贴图资源集构建
### 手工构建
贴图处理流程：
1. 搜集原始图片素材
2. 使用图片处理工具抠图，将物体所在图层另存为新文件，以PNG格式保存（JPG不支持透明度通道，必须存为PNG！！！）

### 利用现有的segmentation数据集

## 数据资源

示例中的铁路轨道数据集来自于：https://universe.roboflow.com/goal-lmqiu/railway-tfabb

### 塑料类漂浮物垃圾开源数据资源列表
- https://universe.roboflow.com/floating-object-detection/floating-object
- https://universe.roboflow.com/floating-detection/floating-f7drg
- https://universe.roboflow.com/yes-h4okd/floating-wastes
- https://universe.roboflow.com/rifatx/-floating-garbage-detection-aeqxx
- https://universe.roboflow.com/jam-tta8l/-floating-trash/dataset/1
- https://universe.roboflow.com/plastic-uwdjt/plastic-p4hdm/dataset/2


### 砖石类异物开源数据资源列表
- https://universe.roboflow.com/yyk-62oza/stone-sgcjn/dataset/5
- https://universe.roboflow.com/project-7y0hj/stone-c3ezu/dataset/1
- https://universe.roboflow.com/klippies/stone-detector-2ecqr/dataset/2

## 更新日志
2025.11.27 整理仓库、上传