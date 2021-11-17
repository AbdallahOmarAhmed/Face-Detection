
import torch
import numpy as np
from albumentations import DualTransform, denormalize_bbox, normalize_bbox
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import albumentations as Aug
import cv2


grid_size = 14
img_size = 448


class CustomCutout(DualTransform):
    def __init__(
            self,
            fill_value=0,
            bbox_removal_threshold=0.6,
            min_size=32,
            max_size=64,
            always_apply=False,
            p=0.5):
        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_cutout_size = min_size
        self.max_cutout_size = max_size

    def get_cutout_position(self, img_height, img_width, cutout_size):
        return (np.random.randint(0, img_width - cutout_size + 1),
                np.random.randint(0, img_height - cutout_size + 1))

    def get_cutout(self, img_height, img_width):
        """
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        cutout_position = self.get_cutout_position(img_height, img_width, cutout_size)
        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position

    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, _ = image.shape
        cutout_arr, cutout_size, cutout_pos = self.get_cutout(self.img_height, self.img_width)

        # Set to instance variables to use this later
        self.image = image
        self.cutout_pos = cutout_pos
        self.cutout_size = cutout_size
        x=0
        y=1
        image[cutout_pos[y]:cutout_pos[y] + cutout_size, cutout_pos[x]:cutout_size + cutout_pos[x], :] = cutout_arr
        return image

    def apply_to_bbox(self, bbox, **params):
        """
        Removes the bounding boxes which are covered by the applied cutout

        :param bbox: A single bounding box coordinates in pascal_voc format
        :returns transformed bbox's coordinates
        """

        # Denormalize the bbox coordinates
        bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
        x_min, y_min, x_max, y_max = tuple(map(int, bbox))

        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
        overlapping_size = np.sum(
            (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value))

        # Remove the bbox if it has more than some threshold of content is inside the cutout patch
        if overlapping_size / max(bbox_size, 1) > self.bbox_removal_threshold:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        return normalize_bbox(bbox, self.img_height, self.img_width)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')

def getAug(train):
    if train:
        transforms = Aug.Compose([
            Aug.RandomResizedCrop(img_size, img_size, scale=(0.5, 0.9)),
            CustomCutout(min_size=100, max_size=200, p=0.8),
            Aug.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30),
            Aug.RandomBrightnessContrast(p=0.4),
            Aug.ToGray(p=0.1),
            Aug.HorizontalFlip(p=0.5),
            Aug.Normalize(),
            ToTensorV2()
        ], bbox_params=Aug.BboxParams(format='pascal_voc', min_visibility=0.33))
        return transforms
    else:
        transforms = Aug.Compose([
            Aug.Resize(img_size, img_size),
            Aug.Normalize(),
            ToTensorV2()
        ], bbox_params=Aug.BboxParams(format='pascal_voc', min_visibility=0.33))
        return transforms

class WiderDataset(Dataset):
    def __init__(self, train=True, max_faces=-1):
        self.train = train
        self.X=[]
        self.Y=[]
        self.path = 'WIDER_train/images/' if train else 'WIDER_val/images/'
        self.txtPath = 'wider_face_split/wider_face_train_bbx_gt.txt' if train\
            else 'wider_face_split/wider_face_val_bbx_gt.txt'
        with open(self.txtPath) as f:
            while True:
                imgName = f.readline().strip('\n')
                if not imgName:
                    break
                c = int(f.readline())
                if c == 0:
                    _ = f.readline()
                    continue
                if c > max_faces and max_faces != -1:
                    for i in range(c):
                        _ = f.readline()
                    continue
                anno=[]
                for i in range(c):
                    y = f.readline().split()
                    y = list(map(float, y))
                    anno.append(y)
                self.X.append(imgName)
                self.Y.append(anno)
        print('finished loading dataset : ', len(self.X))

    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        aug = getAug(self.train)
        v = 1e-9
        y0 = self.Y[index]
        labels = []
        image = cv2.imread(self.path+self.X[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for k in y0:
            l = (k[0], k[1], k[0]+k[2]+v, k[1]+k[3]+v, 1)
            if l[2] <= image.shape[1]+v and l[3] <= image.shape[0]+v:
                labels.append(l)
        labels = np.array(labels)
        transformed = aug(image=image, bboxes=labels)
        x = transformed['image']
        y = transformed['bboxes']
        return x, y


def myCollate(batch):
    images = []
    boxes = []
    for x,y in batch:
        images.append(x)
        boxes.append(y)
    return torch.stack(images), boxes

def draw(frame, face_locations):
    frame = frame.numpy()
    c = 0
    size = img_size / grid_size
    for out in face_locations:
        if out[4] == 1:

            k = list(map(int, out))
            # print(k)
            color = (256, 0, 0)
            cv2.rectangle(frame, (k[0], k[1]), (k[2], k[3]), color, 2)
        c += 1
    cv2.imshow('image', frame)
    cv2.waitKey(0)
