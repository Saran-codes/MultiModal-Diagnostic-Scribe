import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BioGptTokenizer, BioGptForCausalLM
import torchvision.models as models
import wandb 
import signal
import os
import time
import pandas as pd
from tqdm import tqdm
import evaluate 
import numpy as np
import html
import sys
import re


# Local Imports
from model import CytologyVLM, VisionEncoder 
from dataset import Stage2Dataset

def log_research(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open("output.log", "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")

exit_requested = False
def signal_handler(sig, frame):
    global exit_requested
    log_research("\n!!! [STOP SIGNAL] !!! Finishing current batch and shutting down...")
    exit_requested = True

signal.signal(signal.SIGINT, signal_handler)

def log_forensic_table(logits, labels, tokenizer, epoch, step, image_id):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    probs = torch.softmax(shift_logits, dim=-1)
    sample_probs = probs[0] 
    sample_labels = shift_labels[0]

    table = wandb.Table(columns=["Image ID", "Index", "Ground Truth", "Top 1", "Top 2", "Top 3", "Top 4", "Top 5"])

    for i in range(sample_labels.size(0)): 
        target_id = sample_labels[i].item()
        if target_id == -100: continue 

        top_probs, top_ids = torch.topk(sample_probs[i], 5)
        gt_word = tokenizer.decode([target_id]).strip()
        row = [str(image_id), i, gt_word] 
        for p, tid in zip(top_probs, top_ids):
            word = tokenizer.decode([tid]).strip()
            row.append(f"{word} ({p.item()*100:.1f}%)")
        table.add_data(*row)
    wandb.log({f"Forensics/Phase5_Epoch_{epoch}_Step_{step}": table})
    log_research(f"Epoch {epoch} | Step {step}: Forensic Table uploaded.")

def generate_and_log(model, images, full_batch_input_ids, full_batch_labels, tokenizer, device, epoch, step, image_id, config, phase="Val"):
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'): 
            sample_labels = full_batch_labels[0]
            text_labels = sample_labels[256:] 
            valid_indices = (text_labels != -100).nonzero(as_tuple=True)[0]
            first_json_idx = valid_indices[0].item() if len(valid_indices) > 0 else 20
            prompt_ids = full_batch_input_ids[0:1, :first_json_idx]
            
            anchor_text = '{'
            anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False, return_tensors="pt").to(device)
            forced_prompt_ids = torch.cat([prompt_ids, anchor_ids], dim=1)
            prompt_text = tokenizer.decode(forced_prompt_ids[0], skip_special_tokens=True)

            gen_ids = model.generate(
                images, forced_prompt_ids, 
                config['max_seq_len'], tokenizer,
                temperature=0.7, repetition_penalty=1.2
            )
            prompt_length = forced_prompt_ids.shape[1]
            
            gen_text = tokenizer.decode(gen_ids[0][prompt_length:], skip_special_tokens=True)
            full_gen_text = anchor_text + gen_text
            
            log_research(f"--- {phase} Gen | Epoch {epoch} | Step {step} | ID: {image_id} ---")
            log_research(f"PROMPT: {prompt_text}")
            log_research(f"OUTPUT: {full_gen_text}")
            log_research("-" * 50)

            unified_html = f"""
            <div style='font-family: monospace; border: 1px solid #ccc; padding: 15px; background: #fafafa;'>
            <h3 style='margin-top: 0; color: #333;'>Phase 5 | {phase} | Epoch: {epoch} | Step: {step} | ID: {image_id}</h3>
            <div style='background: #e9ecef; padding: 10px; border-left: 4px solid #6c757d; margin-bottom: 10px;'>
            <b>PROMPT:</b> {html.escape(prompt_text)}
            </div>
            <div style='background: #e2f0d9; padding: 10px; border-left: 4px solid #28a745;'>
            <b>MODEL OUTPUT:</b> {html.escape(full_gen_text)}
            </div>
            </div>
            """
            wandb.log({f"Phase5_Gen/{phase}_Step_{step}": wandb.Html(unified_html)})
            return full_gen_text

def clean_format(text):
    """Fixes BioGPT underscore-spacing artifacts before metric calculation."""
    text = re.sub(r'specimen _ type', 'specimen_type', text)
    text = re.sub(r'specimen _ adequacy', 'specimen_adequacy', text)
    text = re.sub(r'general _ categorization', 'general_categorization', text)
    text = re.sub(r'microscopic _ description', 'microscopic_description', text)
    return text

def train(model, train_loader, val_loader, optimizer, scheduler, tokenizer, config, train_df, start_epoch=0):
    wandb.init(
        project="Cytology-VLM-Thesis",
        config=config,
        name="Phase5_DeepVision_Unfrozen",
        id=config.get("wandb_run_id"),
        resume="must"
    )

    val_iter = iter(val_loader)
    static_val_batch = next(val_iter)
    val_imgs = static_val_batch['image'].to(config['device'])
    val_img_ids = static_val_batch['img_id']
    val_input_ids = static_val_batch['input_ids'].to(config['device'])
    val_labels = static_val_batch['labels'].to(config['device'])

    accumulation_steps = config.get('accumulation_steps', 1)
    scaler = torch.amp.GradScaler('cuda') 
    optimizer.zero_grad() 
    best_val_loss = float('inf')

    LABEL_MAP = {"NILM": 0, "LSIL": 1, "HSIL": 2, "ADENO": 3, "SCC": 4}
    mapped_ints = train_df['mapped_label'].astype(str).str.upper().map(LABEL_MAP)
    counts = mapped_ints.value_counts().sort_index().values
    total_samples = counts.sum()
    base_weights = total_samples / (5 * counts)
    final_weights = torch.tensor(base_weights, dtype=torch.float).to(config['device'])

    train_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1, reduction='none')
    val_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            if exit_requested: break

            imgs = batch['image'].to(config['device'])
            image_ids = batch['img_id'] 
            input_ids = batch['input_ids'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            labels_int = batch['label_int'].to(config['device'])

            with torch.amp.autocast('cuda'):
                logits = model(imgs, input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                raw_token_loss = train_criterion(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                ).view(shift_labels.shape)
                batch_weights = final_weights[labels_int].unsqueeze(1) 
                weighted_loss = (raw_token_loss * batch_weights)[shift_labels != -100].mean()
                raw_loss_val = weighted_loss.item()
                loss = weighted_loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer) 
                current_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad() 

            if step > 0 and step % 250 == 0:
                generate_and_log(model, imgs[0:1], input_ids[0:1], labels[0:1], tokenizer, config['device'], epoch, step, image_ids[0], config, "Train")
                model.train()

            if step > 0 and step % 500 == 0:
                with torch.amp.autocast('cuda'):
                    log_forensic_table(logits, labels, tokenizer, epoch, step, image_ids[0])
                generate_and_log(model, val_imgs[0:1], val_input_ids[0:1], val_labels[0:1], tokenizer, config['device'], epoch, step, val_img_ids[0], config, "Val")
                model.train()

            wandb.log({"Batch/Loss": raw_loss_val, "Batch/LR": scheduler.get_last_lr()[0], "Batch/GradNorm": current_grad_norm})
            pbar.set_postfix(loss=raw_loss_val)

        if exit_requested: break

        val_loss = validate(model, val_loader, tokenizer, config, epoch, val_criterion)
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_checkpoint(model, optimizer, scheduler, epoch, val_loss)
            log_research(f"*** PHASE 5 NEW BEST! ({best_val_loss:.4f}) ***")

    wandb.finish()

def validate(model, loader, tokenizer, config, epoch, val_criterion):
    model.eval()
    total_val_loss = 0
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    all_preds, all_refs = [], []
    
    log_research(f"Starting validation epoch {epoch+1}.")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if exit_requested: break
            imgs, image_ids = batch['image'].to(config['device']), batch['img_id']
            input_ids, labels = batch['input_ids'].to(config['device']), batch['labels'].to(config['device'])

            with torch.amp.autocast('cuda'): 
                logits = model(imgs, input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = val_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_val_loss += loss.item()

            if i < 5: 
                for b_idx in range(imgs.size(0)):
                    text_labels = labels[b_idx][256:]
                    valid_indices = (text_labels != -100).nonzero(as_tuple=True)[0]
                    first_json_idx = valid_indices[0].item() if len(valid_indices) > 0 else 20
                    prompt_ids = input_ids[b_idx:b_idx+1, :first_json_idx]
                    
                    gen_ids = model.generate(imgs[b_idx:b_idx+1], prompt_ids, config['max_seq_len'], tokenizer)
                    
                    # Decoded normally, then cleaned purely for the metric array
                    pred_text = tokenizer.batch_decode(gen_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]
                    all_preds.append(clean_format(pred_text))
                    
                    clean_labels = labels[b_idx:b_idx+1].clone()
                    clean_labels[clean_labels == -100] = tokenizer.pad_token_id
                    ref_text = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)[0]
                    all_refs.append(clean_format(ref_text))

    avg_loss = total_val_loss / len(loader)
    
    # BLEU requires a list of lists for references
    refs_for_bleu = [[ref] for ref in all_refs]
    
    try:
        rouge_res = rouge.compute(predictions=all_preds, references=all_refs)['rougeL']
        bleu_res = bleu.compute(predictions=all_preds, references=refs_for_bleu)['bleu']
    except Exception as e:
        log_research(f"Metric Error: {e}")
        rouge_res, bleu_res = 0.0, 0.0
        
    wandb.log({"Epoch/Val_Loss": avg_loss, "Epoch/ROUGE-L": rouge_res, "Epoch/BLEU": bleu_res})
    log_research(f"PHASE 5 VAL: Loss: {avg_loss:.4f} | ROUGE-L: {rouge_res:.4f} | BLEU: {bleu_res:.4f}")
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss):
    path = f"checkpoints/phase5_latest.pth"
    torch.save({
        'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, path)

def save_best_checkpoint(model, optimizer, scheduler, epoch, loss):
    path = f"checkpoints/best_thesis_model_phase5.pth"
    torch.save({'model_state_dict': model.state_dict(), 'loss': loss}, path)

if __name__ == "__main__":
    config = {
        "device": "cuda:2", "epochs": 10, "batch_size": 8, "lr": 1e-5, "max_seq_len": 1024, 
        "accumulation_steps": 1,
        "train_csv": "./data/train_split.csv", "val_csv": "./data/val_split.csv", 
        "img_dir": "./data/images", "report_dir": "./data/reports",
        "wandb_run_id": "ncms3n3k"
    }

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    llm = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    vision_encoder = VisionEncoder()

    model = CytologyVLM(vision_encoder, llm).to(config['device'])
    model.unfreeze_for_alignment(num_layers=4, unfreeze_vision=True)

    train_df = pd.read_csv(config['train_csv'])
    val_df = pd.read_csv(config['val_csv'])

    train_loader = DataLoader(Stage2Dataset(train_df, config['img_dir'], config['report_dir'], tokenizer, max_seq_len=config['max_seq_len'], is_train=True), 
                              batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(Stage2Dataset(val_df, config['img_dir'], config['report_dir'], tokenizer, max_seq_len=config['max_seq_len'], is_train=False), 
                            batch_size=config['batch_size'], shuffle=False, num_workers=4)

    optimizer = optim.AdamW([
        {'params': model.vision_encoder.parameters(), 'lr': 1e-6}, 
        {'params': model.visual_projection.parameters(), 'lr': 1e-5},
        {'params': [p for n, p in model.llm.named_parameters() if p.requires_grad], 'lr': 1e-5}
    ])
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-6, 1e-5, 1e-5], total_steps=total_steps)

    phase4_path = "checkpoints/phase4_epoch_end_checkpoint.pth"
    phase5_path = "checkpoints/phase5_latest.pth"
    start_epoch = 0

    if os.path.exists(phase5_path):
        log_research(f"Resuming from latest checkpoint.")
        ckpt = torch.load(phase5_path, map_location=config['device'])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        log_research(f"Resumed from epoch {start_epoch}.")
    elif os.path.exists(phase4_path):
        log_research(f"Starting from phase4 weights.")
        ckpt = torch.load(phase4_path, map_location=config['device'])
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = 0
    else:
        log_research("No checkpoints found. Starting from scratch.")
    
    train(model, train_loader, val_loader, optimizer, scheduler, tokenizer, config, train_df, start_epoch=start_epoch)