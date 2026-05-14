import torch
import json
import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import logging

# Set up logging for corrupt images
logging.basicConfig(filename='./failed_images_stage2.log', level=logging.ERROR, 
                    format='%(asctime)s - ID: %(message)s')

LABEL_MAP = {"NILM": 0, "LSIL": 1, "HSIL": 2, "ADENO": 3, "SCC": 4}

FULL_LABEL_MAP = {
    "NILM": "Negative for Intraepithelial Lesion or Malignancy",
    "LSIL": "Low-grade Squamous Intraepithelial Lesion",
    "HSIL": "High-grade Squamous Intraepithelial Lesion",
    "ADENO": "Adenocarcinoma",
    "SCC": "Squamous Cell Carcinoma"
}

# 50/50 Split Prompt Pool
PROMPT_POOL_GUIDED = [
    "Correlate the visual evidence with the clinical diagnosis of {label} to produce a Bethesda JSON report:",
    "Identify the diagnostic criteria for {label} in this image and structure the findings accordingly:",
    "Review the specimen and detail the nuclear and cytoplasmic indicators for a {label} diagnosis:",
    "Using the provided image, document the specific characteristics of this {label} case in JSON:"
]

PROMPT_POOL_UNGUIDED = [
    "Generate a structured Bethesda report for this cytology specimen:",
    "Analyze the visual features in this image and provide findings:",
    "Based on the observed cellular patterns, create a pathology JSON:",
    "Perform an automated Bethesda analysis for the following specimen:"
]

class Stage2Dataset(Dataset):
    def __init__(self, df, img_dir, report_dir, tokenizer, canvas_size=1024, max_seq_len=512, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.tokenizer = tokenizer
        self.canvas_size = canvas_size
        self.max_seq_len = max_seq_len
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
        
        label_str = str(row['mapped_label']).upper()
        label_int = LABEL_MAP.get(label_str, 0)
        full_label_str = FULL_LABEL_MAP.get(label_str, label_str)
        
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
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

            img_tensor = TF.to_tensor(image)
            if self.is_train and image.size[0] > 224 and image.size[1] > 224:
                img_tensor = self.cutout(img_tensor)

            c_w, c_h = image.size 
            canvas = torch.zeros((3, self.canvas_size, self.canvas_size))
            max_x, max_y = self.canvas_size - c_w, self.canvas_size - c_h
            
            if self.is_train:
                x_off, y_off = random.randint(0, max_x), random.randint(0, max_y)
            else:
                x_off, y_off = max_x // 2, max_y // 2
            
            canvas[:, y_off:y_off+c_h, x_off:x_off+c_w] = img_tensor
            image_final = self.normalize(canvas)
            if self.is_train:
                image_final = self.color_jitter(image_final)

            version = random.choice([1, 2])
            try:
                with open(os.path.join(self.report_dir, f"{img_id}_report_{version}.json"), 'r') as f:
                    report_text = json.dumps(json.load(f))
            except:
                with open(os.path.join(self.report_dir, f"{img_id}_report_{1 if version==2 else 2}.json"), 'r') as f:
                    report_text = json.dumps(json.load(f))

            if random.random() < 0.5:
                selected_prompt = random.choice(PROMPT_POOL_GUIDED).format(label=full_label_str)
            else:
                selected_prompt = random.choice(PROMPT_POOL_UNGUIDED)
            
            prompt_ids = self.tokenizer.encode(selected_prompt, add_special_tokens=False)
            report_ids = self.tokenizer.encode(" " + report_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            # Sequence Math: 512 total - 256 (Visual Tokens) = 256 Text budget
            max_text_len = self.max_seq_len - 256 
            input_ids = (prompt_ids + report_ids)[:max_text_len]
            if input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids[-1] = self.tokenizer.eos_token_id
            
            num_prompt_tokens = min(len(prompt_ids), len(input_ids))

            # Masking: Prefix (256 Visual) + Prompt is masked
            labels = ([-100] * 256) + ([-100] * num_prompt_tokens) + input_ids[num_prompt_tokens:]

            # Padding
            padding_text = max_text_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_text
            labels += [-100] * (self.max_seq_len - len(labels))

            return {
                "image": image_final,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "label_int": torch.tensor(label_int, dtype=torch.long),
                "img_id": img_id
            }

        except Exception as e:
            logging.error(f"{img_id} - Error: {str(e)}")
            return {
                "image": torch.zeros((3, self.canvas_size, self.canvas_size)),
                "input_ids": torch.full((self.max_seq_len - 256,), self.tokenizer.pad_token_id, dtype=torch.long),
                "labels": torch.full((self.max_seq_len,), -100, dtype=torch.long),
                "label_int": torch.tensor(-1, dtype=torch.long),
                "img_id": img_id
            }