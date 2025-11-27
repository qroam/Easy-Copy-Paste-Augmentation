# Easy Copy-Paste Augmentation (CV object detection data synthesize)

**中文版说明：** [README_zh.md](https://github.com/qroam/Easy-Copy-Paste-Augmentation/blob/main/docs/README_zh.md)

![Image failed to load](/docs/demo.jpg "Demo")

## Quick Start
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

## Dependencies
```bash
pip3 install -r requirements.txt
```

## Introduction
### Background

In recent years, deep-learning–based computer vision object detection has become the mainstream approach and has been widely deployed across numerous industrial scenarios. 
Deep learning models are data-driven as their training relies heavily on large volumes of data. 
Obtaining abundant and diverse data to cover the various situations that may arise in real-world applications is a vital factor in a DL–based visual detection project. 
Models are often constrained by data bottlenecks and risk poor generalization, greatly limiting the efficiency of industrial-level deployment.

Traditional data collection practice requires a period of time to gather sufficient amount of customized data, followed by manual annotation. Such brute-force approach is labour-intensive and time-consuming. More seriously, in practice, customer do not always have the patience to wait several months for algorithm developers to complete data collection and labeling.

See the following examples. In some manufacturing scenarios, customer may want a model that can detect various tools and equipments to ensure that workers are carrying right ones.
Customer might provide just a list of tools, along with a single reference image, while expect algorithm developers to produce a demonstrable model. 
It may sound *unreasonable* if algorithm developers say something like ``We have to install cameras in the factory and conduct months of on-site image collection''.
In other cases, real samples are extremely difficult to obtain. For instance, in scenarios involving anomalies, defects, or risks, where most real-world samples are normal, resulting in severe sample scarcity and class imbalance. 
For example, in railway transportation, one may wish to detect foreign objects such as stones or plastic debris on the tracks to mitigate safety hazards. If we rely solely on real sample collection, we would essentially have to place these objects on a test track one by one. But even so, it would be difficult to gather enough data.

In fact, adding objects to images that do not originally contain them is something people have wanted to do for a long time. This is why Photoshop became so popular more than a decade ago. In the deep learning community, this idea is not new either. Prior research [Ref. 1, 2] has already shown that is a simple yet highly effective method of constructing training data. In the literature, this approach is generally referred to by a common term: **Copy-Paste Augmentation**.

The advantages of copy-paste augmentation come from two-fold: to rapidly generate large quantities of data; and to automatically obtain annotation information from object contour.

We have to admit that recent advances in generative models (such as the widely discussed Nano Banana) make this ancient method appear somewhat outdated. However, several considerations still make copy-paste augmentation have unique practical value: 
1) *computational demands* (generative models usually require notable compute resources and offer slower inference, with throughput far lower than copy-paste augmentation), 
2) *adaptation overhead* (e.g., generating domain-specific images may require multiple rounds of trial-and-error to pick up a suitable generative model), 
and 
3) *the gap between generated samples and annotated samples* (generative models can not output annotated labels alongside the output image, although it is viable to composite generative models ith general-purpose inference models like SAM to build automated data-labeling pipeline, or rely on conditional generation).

In certain scenarios, the objects are *highly self-contained*, meaning that their semantics do not change when placed into a new context. Such objects, e.g. foreign objects, people, and animals, are naturally well-suited for data construction via copy-paste augmentation. 
Conversely, for objects whose recognition relies heavily on contextual information, we recommend using generative models to construct training samples.


### Motivation and Features of this Repo

Despite its widely adoptation in industry, there are currently few user-friendly open-source implementations of Copy-Paste Augmentation. 
Existing libraries are often either too minimalistic or heavily encapsulated within large frameworks, making them inconvenient to extend.

For example, Ultralytics YOLO includes a built-in copy-paste augmentation method [Ref. 5], but it only supports pasting annotated instance-segmentation masks between existing training samples. 
It cannot address the problem of generating samples ``from scratch''. And because the generated images are directly used for training, it is difficult to apply quality control.

To address these limitations, this Repo provides an implementation of copy-paste augmentation that offers *four key advantages*, enabling more fine-grained control over the generation of realistic synthetic data:

1. **Pasting Rules**: Supports three major rules—position, scale, and angle—to control how foreground objects are pasted onto background images.

2. **Configuration-Based Workflow**: Supports (and strongly recommends) specifying data-generation rules through an `yaml` configuration file. 
This ensures a clear directory structure, flexible and controllable workflows, and the configuration file itself serves as a log for future verification.

3. **Conditional Pasting**: Within the pasting rules, it supports using existing annotations in the background image (in LabelMe format) as conditional information. 
For example, the placement position or scaling of the pasted object can refer to objects already present in the image.

4. **Optimization on Copy-Paste Quality**: Supports feathering and edge blurring to make foreground objects fuse more naturally with the background. This can prevent model from overfitting low-level artifacts such as jagged cut edges.


<!-- 贴入负例， -->


### References
1. Dwibedi et al., 2017. Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection. https://arxiv.org/pdf/1708.01642
https://github.com/debidatta/syndata-generation

2. Ghiasi et al., 2021. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. https://arxiv.org/abs/2012.07177v1

3. Zhao et al., 2023. X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion. https://proceedings.mlr.press/v202/zhao23f/zhao23f.pdf https://github.com/yoctta/XPaste

4. https://docs.ultralytics.com/guides/yolo-data-augmentation/#copy-paste-copy_paste

5. https://github.com/conradry/copy-paste-aug

6. https://github.com/AICVHub/Copy-Paste-for-Semantic-Segmentation

7. https://github.com/MarkPotanin/copy_paste_aug_detectron2

8. https://github.com/Opletts/Copy-Paste-Augmenter



## Copy-Paste Rules

The design consideration of Copy-Paste Rules is to control the way the foreground object is inserted onto the background image through three sets of parameters: position (x, y), scaling ratio s, and rotation angle r. Also it is important to set how much objects should be inserted using a number rule.

1. number rule：
to control the number of each class of object inserted into one background image.
You can provide a range like `[0, 5]`.

2. position rule：
to control where to place the inserted objects in the background image.
For objects that usually appear in a specific context, set this rule to ensure that the objects are positioned within a specific range or above specific types of objects. 
For objects that can appear randomly, it's okay to not set this rule.

3. scale rule：
to control the size of inserted objects, making them more realistic.
You can set this rule based on the whole background image; or based on specific type of object in the current annotations.

4. angle rule：to control the rotation angle of inserted objects.


## Config File
See `example/foreign_object_v1.yaml` for an example of copy-paste config file we design to indicate the previous rules.


## Construction of foreground mask dataset
### Manual labour
1. Collect original image materials for foreground objects.
2. Use image processing tools to *cut* objects from background, save the layer containing the object into a new file. **Remember to save it in PNG format since JPG does not support a transparency channel.**

### Reusing Segmentation Datasets

## Data Resources

The railway track dataset in the example is from: https://universe.roboflow.com/goal-lmqiu/railway-tfabb

### A list of plastic garbage datasets:
- https://universe.roboflow.com/floating-object-detection/floating-object
- https://universe.roboflow.com/floating-detection/floating-f7drg
- https://universe.roboflow.com/yes-h4okd/floating-wastes
- https://universe.roboflow.com/rifatx/-floating-garbage-detection-aeqxx
- https://universe.roboflow.com/jam-tta8l/-floating-trash/dataset/1
- https://universe.roboflow.com/plastic-uwdjt/plastic-p4hdm/dataset/2

### A list of stone object datasets:
- https://universe.roboflow.com/yyk-62oza/stone-sgcjn/dataset/5
- https://universe.roboflow.com/project-7y0hj/stone-c3ezu/dataset/1
- https://universe.roboflow.com/klippies/stone-detector-2ecqr/dataset/2

## Update Log
2025.11.27 Create Repo and upload codes