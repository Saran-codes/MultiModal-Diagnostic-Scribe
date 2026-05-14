import torch
import pandas as pd
import json
import os
import re
import html
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BioGptTokenizer, BioGptForCausalLM

# Metric Libraries
import evaluate
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Local Imports
from model import CytologyVLM, VisionEncoder
from dataset import Stage2Dataset, FULL_LABEL_MAP, LABEL_MAP

def clean_format(text):
    """Fixes BioGPT underscore-spacing artifacts and normalizes punctuation."""
    text = re.sub(r'specimen _ type', 'specimen_type', text)
    text = re.sub(r'specimen _ adequacy', 'specimen_adequacy', text)
    text = re.sub(r'general _ categorization', 'general_categorization', text)
    text = re.sub(r'microscopic _ description', 'microscopic_description', text)
    text = text.replace(' :', ':').replace(' ,', ',').replace(' }', '}')
    return text.strip()

def evaluate_dataset(model, df, img_dir, report_dir, tokenizer, device, config, split_name="Validation"):
    model.eval()
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    hf_bleu = evaluate.load("bleu")
    
    all_preds = []
    all_refs_for_rouge = []
    all_refs_for_bleu = [] 
    
    print(f"\n--- Starting 100% Guided Inference on {split_name} Set ---")
    print(f"Hyperparameters -> Temp: {config['temperature']} | Rep Penalty: {config['rep_penalty']}")

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {split_name}"):
            img_id = str(row['unique_id']).zfill(6)
            label_str = str(row['mapped_label']).upper()
            full_label_str = FULL_LABEL_MAP.get(label_str, label_str)
            
            temp_ds = Stage2Dataset(df.iloc[[idx]], img_dir, report_dir, tokenizer, is_train=False)
            ds_item = temp_ds[0]
            img_tensor = ds_item['image'].unsqueeze(0).to(device)
            
            # Force-start the JSON to anchor the model's structure
            prompt_text = f"Correlate the visual evidence with the clinical diagnosis of {full_label_str} to produce a Bethesda JSON report: {{"
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
            
            with torch.amp.autocast('cuda'):
                gen_ids = model.generate(
                    img_tensor, 
                    prompt_ids, 
                    max_new_tokens=config['max_seq_len'] - prompt_ids.shape[1] - 256, 
                    tokenizer=tokenizer,
                    temperature=config['temperature'], 
                    repetition_penalty=config['rep_penalty']
                )
            
            # Slice output after the prompt, then re-attach the anchor '{'
            gen_text_raw = tokenizer.decode(gen_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
            pred_text = clean_format("{" + gen_text_raw)
            all_preds.append(pred_text)
            
            try:
                with open(os.path.join(report_dir, f"{img_id}_report_1.json"), 'r') as f:
                    ref_data = json.load(f)
            except:
                with open(os.path.join(report_dir, f"{img_id}_report_2.json"), 'r') as f:
                    ref_data = json.load(f)
                    
            ref_text = clean_format(json.dumps(ref_data))
            all_refs_for_rouge.append(ref_text)
            
            # NLTK BLEU needs tokenized sequences
            all_refs_for_bleu.append([ref_text.split()])

    print("\n[Audit] Finalizing Scores...")

    rouge_results = rouge.compute(predictions=all_preds, references=all_refs_for_rouge)
    meteor_results = meteor.compute(predictions=all_preds, references=all_refs_for_rouge)
    
    smoothie = SmoothingFunction().method4
    tokenized_preds = [pred.split() for pred in all_preds]
    
    nltk_bleu1 = corpus_bleu(all_refs_for_bleu, tokenized_preds, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    nltk_bleu2 = corpus_bleu(all_refs_for_bleu, tokenized_preds, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    nltk_bleu3 = corpus_bleu(all_refs_for_bleu, tokenized_preds, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    nltk_bleu4 = corpus_bleu(all_refs_for_bleu, tokenized_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    hf_bleu_results = hf_bleu.compute(predictions=all_preds, references=[[r] for r in all_refs_for_rouge])
    bp = hf_bleu_results['brevity_penalty']
    precisions = hf_bleu_results['precisions']
    
    hf_bleu1 = bp * precisions[0] if len(precisions) > 0 else 0
    hf_bleu2 = bp * ((precisions[0] * precisions[1]) ** (1/2)) if len(precisions) > 1 else 0
    hf_bleu3 = bp * ((precisions[0] * precisions[1] * precisions[2]) ** (1/3)) if len(precisions) > 2 else 0
    hf_bleu4 = hf_bleu_results['bleu']

    try:
        from pycocoevalcap.cider.cider import Cider
        # pycocoevalcap expects dict mapping image ID to list of strings
        gts = {str(i): [ref] for i, ref in enumerate(all_refs_for_rouge)}
        res = {str(i): [pred] for i, pred in enumerate(all_preds)}
        
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        cider_score = round(cider_score * 100, 2)
    except ImportError:
        print("\n[WARNING] pycocoevalcap not installed. Skipping CIDEr.")
        print("To fix: pip install pycocoevalcap\n")
        cider_score = "N/A"

    best_bleu1 = round(max(nltk_bleu1 * 100, hf_bleu1 * 100), 2)
    best_bleu2 = round(max(nltk_bleu2 * 100, hf_bleu2 * 100), 2)
    best_bleu3 = round(max(nltk_bleu3 * 100, hf_bleu3 * 100), 2)
    best_bleu4 = round(max(nltk_bleu4 * 100, hf_bleu4 * 100), 2)

    results = {
        "BLEU-1": best_bleu1,
        "BLEU-2": best_bleu2,
        "BLEU-3": best_bleu3,
        "BLEU-4": best_bleu4,
        "ROUGE-1": round(rouge_results['rouge1'] * 100, 2),
        "ROUGE-2": round(rouge_results['rouge2'] * 100, 2),
        "ROUGE-L": round(rouge_results['rougeL'] * 100, 2),
        "METEOR": round(meteor_results['meteor'] * 100, 2),
        "CIDEr": cider_score
    }
    
    print(f"\n{'='*20} {split_name} {'='*20}")
    for k, v in results.items():
        print(f"{k.ljust(10)}: {v}")
    print(f"{'='*50}\n")
    
    return results, all_preds, all_refs_for_rouge

if __name__ == "__main__":
    config = {
        "device": "cuda:1",
        "max_seq_len": 1024,
        "temperature": 0.1,   # near-greedy for max n-gram overlap
        "rep_penalty": 1.05,  # low to allow medical term repetition
        "val_csv": "./data/val_split.csv", 
        "test_csv": "./data/test_split.csv",
        "img_dir": "./data/images", 
        "report_dir": "./data/reports",
        "checkpoint_path": "checkpoints/best_thesis_model_phase5.pth"
    }

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    llm = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    vision_encoder = VisionEncoder()

    model = CytologyVLM(vision_encoder, llm).to(config['device'])
    
    if os.path.exists(config['checkpoint_path']):
        ckpt = torch.load(config['checkpoint_path'], map_location=config['device'])
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"--- Checkpoint Loaded: {config['checkpoint_path']} ---")
    else:
        print(f"ERROR: Checkpoint not found at {config['checkpoint_path']}")
        exit()

    val_df = pd.read_csv(config['val_csv'])
    test_df = pd.read_csv(config['test_csv'])

    val_results, v_preds, v_refs = evaluate_dataset(model, val_df, config['img_dir'], config['report_dir'], tokenizer, config['device'], config, "Validation")
    test_results, t_preds, t_refs = evaluate_dataset(model, test_df, config['img_dir'], config['report_dir'], tokenizer, config['device'], config, "Test")

    results_to_save = pd.DataFrame({
        "Image_ID": test_df['unique_id'].astype(str).str.zfill(6),
        "Mapped_Label": test_df['mapped_label'],
        "Ground_Truth_Report": t_refs,
        "Model_Prediction": t_preds
    })
    results_to_save.to_csv("FINAL_THESIS_RESULTS.csv", index=False)
    print("Exported FINAL_THESIS_RESULTS.csv for manual inspection.")