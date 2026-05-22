"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import faster_coco_eval.core.mask as coco_mask
from faster_coco_eval.utils.pytorch import FasterCocoDetection
import torch
import torchvision
import os
from PIL import Image
import torch.nn.functional as F

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None

import numpy as np
import cv2

__all__ = ["CocoDetection"]


@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = [
        "transforms",
    ]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False
    ):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        image_path = os.path.join(self.img_folder, self.coco.loadImgs(image_id)[0]["file_name"])
        target = {"image_id": image_id, "image_path": image_path, "annotations": target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(
                target["boxes"], key="boxes", spatial_size=image.size[::-1]
            )

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        if hasattr(self, "_preset") and self._preset is not None:
            s += f" preset:\n   {repr(self._preset)}"
        return s

    @property
    def categories(
        self,
    ):
        return self.coco.dataset["categories"]

    @property
    def category2name(
        self,
    ):
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(
        self,
    ):
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(
        self,
    ):
        return {i: cat["id"] for i, cat in enumerate(self.categories)}


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

@register()
class CocoDetectionRGBD(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, depth_folder, **kwargs):
        # We pass transforms to super so self._transforms is set
        super().__init__(img_folder, ann_file, transforms=transforms, **kwargs)
        self.depth_folder = depth_folder

    def load_item(self, idx):
        image, target = super().load_item(idx) 
        rgb_np = np.array(image).astype(np.float32) / 255.0 # Scale RGB to 0-1
        
        file_name = self.coco.loadImgs(self.ids[idx])[0]["file_name"]
        # Use os.path.basename to get '1.jpg' and discard 'images/'
        pure_file_name = os.path.basename(file_name)
        depth_name = pure_file_name.replace('.jpg', '.png')

        depth_path = os.path.join(self.depth_folder, depth_name)

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        
        # Load 16-bit depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (rgb_np.shape[1], rgb_np.shape[0]))
        
        # Scale 16-bit (0-65535) to 0.0-1.0
        depth_np = depth.astype(np.float32) / 65535.0
        depth_np = np.expand_dims(depth_np, axis=-1)
        
        # Now both are 0-1, stack them
        rgbd = np.concatenate([rgb_np, depth_np], axis=-1)
        return rgbd, target

    def __getitem__(self, idx):
            img_np, target = self.load_item(idx) 
            
            # 1. Convert to standard Tensor format [4, H, W]
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()

            # 2. Extract database image metadata directly to align coordinate scales
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            
            json_w = img_info['width']   # 1280
            json_h = img_info['height']  # 720


            LABEL_MAP = {
                1: 1,  # Ball -> Maps to 1
                2: 2,  # Goalpost -> Maps to 2
                3: 3,  # K1 Robot -> Maps to 3
                4: 4,  # L-Intersection -> Maps to 4
                5: 5,  # Penalty Mark -> Maps to 5
                6: 6,  # T-Intersection -> Maps to 6
                7: 7   # X-Intersection -> Maps to 7
            }

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            raw_anns = self.coco.loadAnns(ann_ids)
            
            boxes = []
            labels = []
            
            for ann in raw_anns:
                if ann.get('iscrowd', 0):
                    continue
                    
                raw_box = ann['bbox'] # [xmin, ymin, width, height]
                xmin, ymin, bw, bh = raw_box[0], raw_box[1], raw_box[2], raw_box[3]
                
                # 3. Calculate absolute centers
                cx = xmin + (bw / 2.0)
                cy = ymin + (bh / 2.0)
                
                # 4. Normalize strictly using the JSON file's design space
                cx_norm = cx / json_w
                cy_norm = cy / json_h
                bw_norm = bw / json_w
                bh_norm = bh / json_h

                boxes.append([cx_norm, cy_norm, bw_norm, bh_norm])
                
                # Translate safely using our shifted dictionary map
                raw_category_id = int(ann['category_id'])
                if raw_category_id in LABEL_MAP:
                    labels.append(LABEL_MAP[raw_category_id])
                else:
                    continue

            # --- SANITATION FILTER AT THE END OF GETITEM ---
            if len(boxes) > 0:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                
                # 1. Force strict clipping between 0.0 and 1.0 to eliminate floating point runaways
                boxes_tensor = torch.clamp(boxes_tensor, 0.0, 1.0)
                
                # 2. Filter out any degenerate boxes (Width or Height too close to 0)
                # box structure is [cx, cy, w, h], so indices 2 and 3 are width and height
                valid_mask = (boxes_tensor[:, 2] > 1e-4) & (boxes_tensor[:, 3] > 1e-4)
                
                # 3. Check for any corrupted NaN or Inf values
                finite_mask = torch.isfinite(boxes_tensor).all(dim=-1)
                
                # Combine filters
                keep_mask = valid_mask & finite_mask
                
                cleaned_boxes = boxes_tensor[keep_mask]
                cleaned_labels = labels_tensor[keep_mask]
                
                # 4. Handle the edge case where all boxes in this specific image were invalid
                if len(cleaned_boxes) > 0:
                    target["boxes"] = cleaned_boxes
                    target["labels"] = cleaned_labels
                else:
                    # DETR safety fallback: provide a tiny micro-box instead of an absolute 0-element tensor
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    target["labels"] = torch.zeros((0,), dtype=torch.long)
            else:
                # Safe placeholder for completely empty background images
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.long)

            img = F.interpolate(
                img.unsqueeze(0),  # [1, 4, H, W]
                size=(640, 640),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [4, 640, 640]

            target["orig_size"] = torch.tensor([json_h, json_w], dtype=torch.long)  # [720, 1280]

            return img, target

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        image_path = target["image_path"]

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get("category2label", None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

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
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["image_path"] = image_path
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
