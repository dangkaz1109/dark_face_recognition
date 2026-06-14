import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.model import FacenetBaseline
from src.loss import StandardTripletLoss
from src.dataset import RealFaceTripletDataset

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def apply_camera_noise(img_tensor, p=19.50/255.0, g=38.25, c=0.05):
    img_255 = (img_tensor + 1.0) * 127.5
    a = torch.abs(torch.randn(img_tensor.size(0), 1, 1, 1, device=img_tensor.device) * p)
    b = torch.abs(torch.randn(img_tensor.size(0), 1, 1, 1, device=img_tensor.device) * g)
    sd = torch.abs(a * img_255 + b)
    noisy_255 = img_255 + torch.randn_like(img_255) * sd
    
    red_gain = torch.randn(img_tensor.size(0), 1, 1, device=img_tensor.device) * c + 1.0
    green_gain = torch.randn(img_tensor.size(0), 1, 1, device=img_tensor.device) * c + 1.0
    blue_gain = torch.randn(img_tensor.size(0), 1, 1, device=img_tensor.device) * c + 1.0
    
    noisy_255[:, 0] = noisy_255[:, 0] / red_gain
    noisy_255[:, 1] = noisy_255[:, 1] / green_gain
    noisy_255[:, 2] = noisy_255[:, 2] / blue_gain
    
    noisy_255 = torch.clamp(noisy_255, 0.0, 255.0)
    return (noisy_255 / 127.5) - 1.0


def train():
    config = load_config()
    torch.manual_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_with_noise = config["training"].get("train_with_noise", False)
    print(f"Training configuration: train_with_noise={train_with_noise}")
    
    print("Initializing FacenetBaseline model...")
    model = FacenetBaseline(
        pretrained=config["model"]["backbone_pretrained"]
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    criterion = StandardTripletLoss(margin=1.0)

    print("Loading real dataset...")
    train_dataset = RealFaceTripletDataset(
        csv_path=config["data"]["csv_path"],
        img_dir=config["data"]["img_dir"],
        img_size=config["data"]["img_size"],
        max_triplets_per_class=config["data"]["max_pairs_per_class"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True
    )
    
    print(f"Starting training on {len(train_dataset)} triplets...")
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        
        for img_a, img_p, img_n in train_loader:
            img_a, img_p, img_n = img_a.to(device), img_p.to(device), img_n.to(device)
            
            if train_with_noise:
                img_a = apply_camera_noise(img_a)
                img_p = apply_camera_noise(img_p)
                img_n = apply_camera_noise(img_n)
                
            emb_a = model(img_a)
            emb_p = model(img_p)
            emb_n = model(img_n)
            
            loss = criterion(emb_a, emb_p, emb_n)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * img_a.size(0)
            
        avg_loss = epoch_loss / (len(train_loader) * config["training"]["batch_size"])
        
        # Fast evaluation on a batch of noisy data
        model.eval()
        val_loss_noisy = 0.0
        with torch.no_grad():
            for img_a, img_p, img_n in train_loader:
                img_a, img_p, img_n = img_a.to(device), img_p.to(device), img_n.to(device)
                img_a_noisy = apply_camera_noise(img_a)
                img_p_noisy = apply_camera_noise(img_p)
                img_n_noisy = apply_camera_noise(img_n)
                
                emb_a = model(img_a_noisy)
                emb_p = model(img_p_noisy)
                emb_n = model(img_n_noisy)
                
                loss_val = criterion(emb_a, emb_p, emb_n)
                val_loss_noisy = loss_val.item()
                break # Evaluate on a single validation batch for speed
                
        print(f"Epoch {epoch}/{config['training']['epochs']} | Train Loss: {avg_loss:.4f} | Val Loss (Noisy): {val_loss_noisy:.4f}")
              
    os.makedirs("models", exist_ok=True)
    save_filename = "facenet_noisy.pth" if train_with_noise else "facenet_clean.pth"
    save_path = os.path.join("models", save_filename)
    torch.save(model.state_dict(), save_path)
    print(f"Successfully trained model. Saved state dict to {save_path}")

if __name__ == "__main__":
    train()




