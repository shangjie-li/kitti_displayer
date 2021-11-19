import matplotlib.pyplot as plt # For WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import sys
import os
import numpy as np

import torch
import torch.utils.data as data
import random
import argparse
from pathlib import Path

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='KITTI Displayer')
    parser.add_argument('--images_root', default='images_and_labels', type=str,
                        help='Root directory path to images.')
    parser.add_argument('--info_file', default='instances_train.json', type=str,
                        help='Annotation file (coco form).')
    parser.add_argument('--classes_file', default=None, type=str,
                        help='A file that provides a list of class names.')

    global args
    args = parser.parse_args(argv)

def create_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    color = (r, g, b)
    return color

def draw_mask(img, mask, color):
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0
    
    mask = mask[:, :, None]
    color_tensor = torch.Tensor(color).to(img_gpu.device.index).float() / 255.
    alpha = 0.45
    
    mask_color = mask.repeat(1, 1, 3) * color_tensor * alpha
    inv_alph_mask = mask * (- alpha) + 1
    img_gpu = img_gpu * inv_alph_mask + mask_color
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    return img_numpy

def draw_annotation(img, mask, box, classname, color, score=None):
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    font_thickness, line_thickness = 1, 2
    
    x1, y1, x2, y2 = box[:]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    
    img = draw_mask(img, mask, color)
    
    u, v = int(x1), int(y1)
    text_str = '%s: %.2f' % (classname, score) if score else classname
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    
    if v - text_h - 4 < 0: v = text_h + 4
    cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img

def get_label_map():
    return {x+1: x+1 for x in range(len(specified_classes))}

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res

class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file,
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO

        self.root = image_path
        self.coco = COCO(info_file)
        
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        self.target_transform = COCOAnnotationTransform()
        
        self.name = dataset_name
        self.has_gt = has_gt

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        img_id = self.ids[index]
        
        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]
        
        path = os.path.join(self.root, file_name)
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = cv2.imread(path) # BGR image
        height, width, _ = img.shape
        return img_id, file_name, img, height, width

    def pull_anno(self, index, height, width):
        img_id = self.ids[index]
        
        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = []
        
        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects, height, width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)
            target = np.array(self.target_transform(target, width, height))
        else:
            masks = None
            target = np.array([])
        return img_id, masks, target, num_crowds

if __name__ == '__main__':
    parse_args()
    print('Root directory path to images:\n %s' % Path(args.images_root).resolve())
    print('Annotation file (coco form):\n %s' % Path(args.info_file).resolve())
    
    if args.classes_file:
        path = args.classes_file
        assert os.path.exists(path), 'File does not exist: {}'.format(path)
        specified_classes = []
        for line in open(os.path.join(path)):
            if not line.isspace():
                name = line.strip()
                if '\\t' in name: name = ' '.join(name.split('\\t'))
                specified_classes.append(name)
    else:
        print('Error: The parameter classes_file must be set.')
        exit()
    print('\nSpecified class names:\n', specified_classes)
    
    print()
    
    dataset = COCODetection(args.images_root, args.info_file)

    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i + 1, len(dataset)))
        
        print('Original Data:')
        img_id_i, file_name, img, height, width = dataset.pull_image(i) # BGR
        img_id_a, masks, target, num_crowds = dataset.pull_anno(i, height, width)
        
        print(' image_id:', img_id_i, ' file_name:', file_name)
        print(' img:', type(img), img.shape, ' h: %s w: %s' % (height, width), '\n', img)
        print(' masks:', type(masks), masks.shape, '\n', masks)
        
        boxes = target[:, :4]
        labels = target[:, 4]
        
        print(' boxes:', type(boxes), boxes.shape,'\n', boxes)
        print(' labels:', type(labels), labels.shape,'\n', labels)
        print(' num_crowds:', type(num_crowds), '\n', num_crowds)
        
        img_annotated = img.copy()
        scale = [width, height, width, height]
        
        for j in range(boxes.shape[0]):
            mask = torch.from_numpy(masks[j]).cuda().float()
            box = boxes[j] * scale
            label = int(labels[j])
            if label >= 0:
                classname = specified_classes[label]
            else:
                classname = 'crowd'
            
            color = create_random_color()
            img_annotated = draw_annotation(img_annotated, mask, box, classname, color)
        cv2.imshow('Original Data', img_annotated)
        
        # press 'Esc' to shut down, and every key else to continue
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            continue
