import os
import csv
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class RealFaceTripletDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, img_size: int = 160, max_triplets_per_class: int = 50, seed: int = 42):
        super(RealFaceTripletDataset, self).__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        identity_to_imgs = {}
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found at: {img_dir}")
            
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['id']
                label = row['label']
                img_path = os.path.join(img_dir, img_name)
                
                if os.path.exists(img_path):
                    if label not in identity_to_imgs:
                        identity_to_imgs[label] = []
                    identity_to_imgs[label].append(img_path)
                    
        random.seed(seed)
        self.triplets = []
        identities = list(identity_to_imgs.keys())
        
        for label, img_paths in identity_to_imgs.items():
            if len(img_paths) < 2:
                continue
            
            class_triplets = []
            for i in range(len(img_paths)):
                for j in range(len(img_paths)):
                    if i == j:
                        continue
                    
                    neg_label = random.choice(identities)
                    while neg_label == label or len(identity_to_imgs[neg_label]) == 0:
                        neg_label = random.choice(identities)
                        
                    neg_img_path = random.choice(identity_to_imgs[neg_label])
                    class_triplets.append((img_paths[i], img_paths[j], neg_img_path))
                    
            if len(class_triplets) > max_triplets_per_class:
                class_triplets = random.sample(class_triplets, max_triplets_per_class)
                
            self.triplets.extend(class_triplets)
            
        print(f"Loaded {len(identity_to_imgs)} identities from {csv_path}.")
        print(f"Generated {len(self.triplets)} triplets for training.")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        img_path_a, img_path_p, img_path_n = self.triplets[idx]
        
        img_a = Image.open(img_path_a).convert('RGB')
        img_p = Image.open(img_path_p).convert('RGB')
        img_n = Image.open(img_path_n).convert('RGB')
        
        return self.transform(img_a), self.transform(img_p), self.transform(img_n)


