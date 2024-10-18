"""
在RT-DETR的基础上进行调整，能够处理并读取红外和可见光两类图像数据
"""

import torch
import torch.utils.data
import torchvision
torchvision.disable_beta_transforms_warning()
import numpy as np
from torchvision import datapoints

from pycocotools import mask as coco_mask
from src.core import register
import os
from PIL import Image
from torchvision import transforms

__all__ = ['CocoDetection']



@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        self.img_ir_folder = os.path.join(img_folder, 'infrared/')
        img_folder = os.path.join(img_folder, 'visible/')
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.img_folder = img_folder
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.ir_paths = [os.path.join(self.img_ir_folder + file) for file in sorted(os.listdir(self.img_ir_folder))]
        self.rgb_paths = [os.path.join(self.img_folder + file) for file in sorted(os.listdir(self.img_folder))]
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img_rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        img_ir = Image.open(self.ir_paths[idx]).convert("RGB")
        img_rgb, img_ir, target = self.prepare(img_rgb, img_ir, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img_rgb.size[::-1])  # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])
        if self._transforms is not None:
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            torch.manual_seed(seed)  # apply this seed to img tranfsorms
            img_rgb, target_rgb = self._transforms(img_rgb, target)
            torch.manual_seed(seed)  # apply this seed to target tranfsorms
            img_ir, target_ir = self._transforms(img_ir, target)

        if self.root.split('/')[-3] == 'train':
            img_rgb, img_ir, bboxes = random_flip_horizon(img_rgb, img_ir, target_ir['boxes'])
            target_ir['boxes'] = bboxes

        return torch.cat([img_rgb, img_ir], dim=0), target_ir


    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s


def random_flip_horizon(img_rgb, img_ir, boxes, h_rate=0.5):
    if np.random.random() < h_rate:
        transform = transforms.RandomHorizontalFlip(p=1)
        img_rgb = transform(img_rgb)
        img_ir = transform(img_ir)
        if len(boxes) > 0:
            x = 1 - boxes[:, 0]
            boxes[:, 0] = x
    if np.random.random() < h_rate:
        transform = transforms.RandomVerticalFlip(p=1)
        img_rgb = transform(img_rgb)
        img_ir = transform(img_ir)
        if len(boxes) > 0:
            y = 1 - boxes[:, 1]
            boxes[:, 1] = y
    return img_rgb, img_ir, boxes

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image_rgb,image_ir, target):
        w, h = image_rgb.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image_rgb, image_ir, target


mscoco_category2name = {
    1: 'person'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}