import torch
import yaml
import os
from torch.utils.data import DataLoader
from src.model import FacenetBaseline
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


def load_baseline_model(config_path: str = "configs/config.yaml", 
                        weights_path: str = None, 
                        device: torch.device = torch.device('cpu')) -> FacenetBaseline:
    config = load_config(config_path)
    model = FacenetBaseline(
        pretrained=config["model"]["backbone_pretrained"]
    )
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded trained baseline model weights from {weights_path}")
        
    model.to(device)
    model.eval()
    return model

def run_inference_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    
    clean_weights = "models/facenet_clean.pth"
    noisy_weights = "models/facenet_noisy.pth"
    
    model_clean = load_baseline_model("configs/config.yaml", clean_weights if os.path.exists(clean_weights) else None, device)
    model_noisy = load_baseline_model("configs/config.yaml", noisy_weights if os.path.exists(noisy_weights) else None, device)
    
    dataset = RealFaceTripletDataset(
        csv_path=config["data"]["csv_path"],
        img_dir=config["data"]["img_dir"],
        img_size=config["data"]["img_size"],
        max_triplets_per_class=config["data"]["max_pairs_per_class"]
    )
    
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    print("\n--- Running Baseline FaceNet Comprehensive Evaluation Demo ---")
    print(f"Total triplets evaluated: {len(dataset)}")
    
    # Trackers for Version 1 (Clean model)
    v1_same_noisy_clean = []
    v1_diff_noisy_clean = []
    v1_same_noisy_noisy = []
    v1_diff_noisy_noisy = []
    
    # Trackers for Version 2 (Noisy model)
    v2_same_noisy_clean = []
    v2_diff_noisy_clean = []
    v2_same_noisy_noisy = []
    v2_diff_noisy_noisy = []
    
    with torch.no_grad():
        for batch_a, batch_p, batch_n in loader:
            batch_a = batch_a.to(device)
            batch_p = batch_p.to(device)
            batch_n = batch_n.to(device)
            
            # Generate noisy batches
            batch_a_noisy = apply_camera_noise(batch_a)
            batch_p_noisy = apply_camera_noise(batch_p)
            batch_n_noisy = apply_camera_noise(batch_n)
            
            # Version 1 Embeddings
            emb_a_c = model_clean(batch_a)
            emb_p_c = model_clean(batch_p)
            emb_n_c = model_clean(batch_n)
            
            emb_a_noisy_c = model_clean(batch_a_noisy)
            emb_p_noisy_c = model_clean(batch_p_noisy)
            emb_n_noisy_c = model_clean(batch_n_noisy)
            
            # Version 1 Distances (Noisy Query vs Clean Reference)
            v1_same_nc = torch.norm(emb_a_c - emb_p_noisy_c, p=2, dim=1).cpu().tolist()
            v1_diff_nc = torch.norm(emb_a_c - emb_n_noisy_c, p=2, dim=1).cpu().tolist()
            v1_same_noisy_clean.extend(v1_same_nc)
            v1_diff_noisy_clean.extend(v1_diff_nc)
            
            # Version 1 Distances (Noisy vs Noisy)
            v1_same_nn = torch.norm(emb_a_noisy_c - emb_p_noisy_c, p=2, dim=1).cpu().tolist()
            v1_diff_nn = torch.norm(emb_a_noisy_c - emb_n_noisy_c, p=2, dim=1).cpu().tolist()
            v1_same_noisy_noisy.extend(v1_same_nn)
            v1_diff_noisy_noisy.extend(v1_diff_nn)
            
            # Version 2 Embeddings
            emb_a_n = model_noisy(batch_a)
            emb_p_n = model_noisy(batch_p)
            emb_n_n = model_noisy(batch_n)
            
            emb_a_noisy_n = model_noisy(batch_a_noisy)
            emb_p_noisy_n = model_noisy(batch_p_noisy)
            emb_n_noisy_n = model_noisy(batch_n_noisy)
            
            # Version 2 Distances (Noisy Query vs Clean Reference)
            v2_same_nc = torch.norm(emb_a_n - emb_p_noisy_n, p=2, dim=1).cpu().tolist()
            v2_diff_nc = torch.norm(emb_a_n - emb_n_noisy_n, p=2, dim=1).cpu().tolist()
            v2_same_noisy_clean.extend(v2_same_nc)
            v2_diff_noisy_clean.extend(v2_diff_nc)
            
            # Version 2 Distances (Noisy vs Noisy)
            v2_same_nn = torch.norm(emb_a_noisy_n - emb_p_noisy_n, p=2, dim=1).cpu().tolist()
            v2_diff_nn = torch.norm(emb_a_noisy_n - emb_n_noisy_n, p=2, dim=1).cpu().tolist()
            v2_same_noisy_noisy.extend(v2_same_nn)
            v2_diff_noisy_noisy.extend(v2_diff_nn)
            
    # Calculate stats helper
    def get_stats(dists):
        if not dists:
            return 0.0, 0.0, 0.0
        return sum(dists)/len(dists), min(dists), max(dists)
        
    v1_snc_avg, v1_snc_min, v1_snc_max = get_stats(v1_same_noisy_clean)
    v1_dnc_avg, v1_dnc_min, v1_dnc_max = get_stats(v1_diff_noisy_clean)
    v1_snn_avg, v1_snn_min, v1_snn_max = get_stats(v1_same_noisy_noisy)
    v1_dnn_avg, v1_dnn_min, v1_dnn_max = get_stats(v1_diff_noisy_noisy)
    
    v2_snc_avg, v2_snc_min, v2_snc_max = get_stats(v2_same_noisy_clean)
    v2_dnc_avg, v2_dnc_min, v2_dnc_max = get_stats(v2_diff_noisy_clean)
    v2_snn_avg, v2_snn_min, v2_snn_max = get_stats(v2_same_noisy_noisy)
    v2_dnn_avg, v2_dnn_min, v2_dnn_max = get_stats(v2_diff_noisy_noisy)
    
    print("\n===========================================================")
    print("=== VERSION 1: Finetuned on CLEAN, Evaluated on NOISY ===")
    print("===========================================================")
    print("1. Noisy Query vs Clean Reference:")
    print(f"   Same identity L2:      avg={v1_snc_avg:.4f} | min={v1_snc_min:.4f} | max={v1_snc_max:.4f}")
    print(f"   Different identity L2: avg={v1_dnc_avg:.4f} | min={v1_dnc_min:.4f} | max={v1_dnc_max:.4f}")
    print(f"   Separation Margin:     {(v1_dnc_avg - v1_snc_avg):.4f}")
    print("2. Noisy Query vs Noisy Reference:")
    print(f"   Same identity L2:      avg={v1_snn_avg:.4f} | min={v1_snn_min:.4f} | max={v1_snn_max:.4f}")
    print(f"   Different identity L2: avg={v1_dnn_avg:.4f} | min={v1_dnn_min:.4f} | max={v1_dnn_max:.4f}")
    print(f"   Separation Margin:     {(v1_dnn_avg - v1_snn_avg):.4f}")
    
    print("\n===========================================================")
    print("=== VERSION 2: Finetuned on NOISY, Evaluated on NOISY ===")
    print("===========================================================")
    print("1. Noisy Query vs Clean Reference:")
    print(f"   Same identity L2:      avg={v2_snc_avg:.4f} | min={v2_snc_min:.4f} | max={v2_snc_max:.4f}")
    print(f"   Different identity L2: avg={v2_dnc_avg:.4f} | min={v2_dnc_min:.4f} | max={v2_dnc_max:.4f}")
    print(f"   Separation Margin:     {(v2_dnc_avg - v2_snc_avg):.4f}")
    print("2. Noisy Query vs Noisy Reference:")
    print(f"   Same identity L2:      avg={v2_snn_avg:.4f} | min={v2_snn_min:.4f} | max={v2_snn_max:.4f}")
    print(f"   Different identity L2: avg={v2_dnn_avg:.4f} | min={v2_dnn_min:.4f} | max={v2_dnn_max:.4f}")
    print(f"   Separation Margin:     {(v2_dnn_avg - v2_snn_avg):.4f}")
    print("===========================================================")
    print("Note: Larger Separation Margin indicates better verification performance.")
    
if __name__ == "__main__":
    run_inference_demo()




