import argparse
import torch
import torch.nn as nn
from transformers import BioGptTokenizer, BioGptForCausalLM
import torchvision.models as models
from torch.utils.data import DataLoader
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Local Imports
from model import CytologyVLM
from dataset import Stage2Dataset

def log_spatial_attention(attentions, input_ids, tokenizer, image_id):
    try:
        # Shape: [Batch, Heads, Seq_Len, Seq_Len] -> [Seq_Len, Seq_Len]
        last_layer_attn = attentions[-1][0].mean(dim=0).detach().cpu().float().numpy()
        
        # 1 label token + 256 visual tokens
        prefix_len = 257
        seq_len = last_layer_attn.shape[0]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Spatial Attention Probe | ID: {image_id}", fontsize=16, fontweight='bold')

        # Targeting early generated words
        target_indices = [prefix_len + 2, prefix_len + 8, prefix_len + 14]

        for i, attn_idx in enumerate(target_indices):
            if attn_idx >= seq_len: 
                axes[i].axis('off')
                continue

            token_id = input_ids[0][attn_idx - prefix_len].item()
            word = tokenizer.decode([token_id]).strip()
            
            # Extract visual token scores (Indices 1 to 256)
            visual_attn = last_layer_attn[attn_idx, 1:prefix_len]
            spatial_grid = visual_attn.reshape(16, 16)

            sns.heatmap(spatial_grid, ax=axes[i], cmap='magma', cbar=True, square=True, 
                        xticklabels=False, yticklabels=False)
            axes[i].set_title(f"Attention for: '{word}'", fontsize=12)

        plt.tight_layout()
        wandb.log({f"Diagnostics/Spatial_Maps_{image_id}": wandb.Image(plt)})
        plt.close()
        print(f"[Probe] Spatial Map for {image_id} uploaded to W&B.")
    except Exception as e:
        print(f"[Probe] Error in spatial mapping: {e}")

def log_top_visual_attentions(attentions, input_ids, tokenizer, image_id):
    try:
        last_layer_attn = attentions[-1][0].mean(dim=0).detach().cpu().float()
        
        table = wandb.Table(columns=[
            "Index", "Word", "Label Token Attn %", "Total Image Attn %",
            "Top 1 Patch", "Top 2 Patch", "Top 3 Patch"
        ])

        prefix_len = 257
        seq_len = last_layer_attn.shape[0]

        for attn_idx in range(prefix_len, seq_len):
            text_idx = attn_idx - prefix_len
            token_id = input_ids[0][text_idx].item()
            
            if token_id == tokenizer.pad_token_id: continue
            word = tokenizer.decode([token_id]).strip()
            if not word: continue

            label_attn_pct = last_layer_attn[attn_idx, 0].item() * 100

            visual_attn_slice = last_layer_attn[attn_idx, 1:prefix_len]
            total_visual_attn_pct = visual_attn_slice.sum().item() * 100

            top_scores, top_ids = torch.topk(visual_attn_slice, 3)

            row = [
                text_idx, 
                word, 
                f"{label_attn_pct:.2f}%", 
                f"{total_visual_attn_pct:.2f}%"
            ]
            
            for score, p_id in zip(top_scores, top_ids):
                row.append(f"Patch {p_id.item()} ({score.item()*100:.2f}%)")
            
            table.add_data(*row)

        wandb.log({f"Diagnostics/Attention_Forensics_{image_id}": table})
        print(f"[Probe] Forensic Table (with Total Image Attn) for {image_id} uploaded.")
    except Exception as e:
        print(f"[Probe] Error in forensic table: {e}")

def run_probe(args):
    wandb.init(project="Cytology-VLM-Thesis", id=args.run_id, resume="must")

    df = pd.read_csv(args.csv_path, dtype={'unique_id': str})
    df['unique_id'] = df['unique_id'].str.strip().str.zfill(6)
    target_id = str(args.image_id).strip().zfill(6)
    target_df = df[df['unique_id'] == target_id]
    
    if target_df.empty:
        print(f"Error: ID {target_id} not found.")
        return

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    llm = BioGptForCausalLM.from_pretrained("microsoft/biogpt", attn_implementation="eager")
    resnet = models.resnet50(weights=None)
    vision_encoder = nn.Sequential(*list(resnet.children())[:-2])

    model = CytologyVLM(vision_encoder, llm, num_labels=6).to(args.device)
    
    print(f"Loading weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dataset = Stage2Dataset(target_df, args.img_dir, args.report_dir, tokenizer, is_train=False)
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))

    imgs = batch['image'].to(args.device)
    input_ids = batch['input_ids'].to(args.device)
    labels_int = batch['label_int'].to(args.device)

    if args.force_null:
        print("--- PROBE MODE: Blind Diagnosis (Label Token = NULL) ---")
        labels_int = torch.full_like(labels_int, 5)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _, attentions = model(imgs, labels_int, input_ids, output_attentions=True)
            log_spatial_attention(attentions, input_ids, tokenizer, target_id)
            log_top_visual_attentions(attentions, input_ids, tokenizer, target_id)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--csv_path", type=str, default="./data/master_registry.csv")
    parser.add_argument("--img_dir", type=str, default="./data/images")
    parser.add_argument("--report_dir", type=str, default="./data/reports")
    parser.add_argument("--force_null", action="store_true")
    
    args = parser.parse_args()
    run_probe(args)