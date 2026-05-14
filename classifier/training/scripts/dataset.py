import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import logging

logging.basicConfig(filename='./failed_images.log', level=logging.ERROR, 
                    format='%(asctime)s - ID: %(message)s')

class CytologyDataset(Dataset):
    def __init__(self, dataframe, image_dir, canvas_size=1024, is_train=True):
        self.df = dataframe
        self.image_dir = image_dir
        self.canvas_size = canvas_size
        self.is_train = is_train
        
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15)
        self.elastic = transforms.ElasticTransform(alpha=50.0, sigma=5.0)
        self.sharpen = transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
        self.cutout = transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['unique_id']).zfill(6)
        label = row['label_int']
        img_path = os.path.join(self.image_dir, f"{img_id}.png")

        try:
            image = Image.open(img_path).convert("RGB")
            
            if self.is_train:
                if max(image.size) > self.canvas_size:
                    crop_w, crop_h = int(image.size[0] * 0.9), int(image.size[1] * 0.9)
                    i, j, h_c, w_c = transforms.RandomCrop.get_params(image, (crop_h, crop_w))
                    image = TF.crop(image, i, j, h_c, w_c)

                angle = random.choice([0, 90, 180, 270])
                if angle != 0: image = TF.rotate(image, angle, expand=True)
                if random.random() > 0.5: image = TF.hflip(image)
                if random.random() > 0.5: image = TF.vflip(image)

            # Passive Scaling
            w, h = image.size
            new_w, new_h = w, h
            if max(w, h) > self.canvas_size:
                scale = self.canvas_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
            if min(new_w, new_h) < 224:
                scale = 224 / min(new_w, new_h)
                new_w, new_h = int(new_w * scale), int(new_h * scale)
                if max(new_w, new_h) > self.canvas_size:
                    final_scale = self.canvas_size / max(new_w, new_h)
                    new_w, new_h = int(new_w * final_scale), int(new_h * final_scale)

            if (new_w, new_h) != (w, h):
                image = image.resize((new_w, new_h), Image.LANCZOS)
            
            if self.is_train:
                image = self.elastic(image)
                image = self.sharpen(image)

            # Tensor Conversion
            img_tensor = TF.to_tensor(image)
            
            if self.is_train:
                img_tensor = self.cutout(img_tensor)

            # Canvas Placement
            c_w, c_h = image.size 
            canvas = torch.zeros((3, self.canvas_size, self.canvas_size))
            max_x, max_y = self.canvas_size - c_w, self.canvas_size - c_h
            
            if self.is_train:
                x_off, y_off = random.randint(0, max_x), random.randint(0, max_y)
            else:
                x_off, y_off = max_x // 2, max_y // 2

            canvas[:, y_off:y_off+c_h, x_off:x_off+c_w] = img_tensor

            if self.is_train:
                canvas = self.color_jitter(canvas)
            
            return self.normalize(canvas), torch.tensor(label, dtype=torch.long), img_id

        except Exception as e:
            logging.error(f"{img_id} - Error: {str(e)}")
            return torch.zeros((3, self.canvas_size, self.canvas_size)), torch.tensor(-1), img_id