import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import os

from dataset import CytologyDataset
from torchvision import models

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV, VAL_CSV, IMAGE_DIR = './train_split.csv', './val_split.csv', './images'
CHECKPOINT_PATH, BEST_MODEL_PATH = './last_checkpoint.pth', './best_model.pth'
LOG_DIR = './runs/cyto_resnet50_v1_phase_c'

BATCH_SIZE = 16 
EPOCHS = 200
MAX_LR = 5e-6

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def get_weighted_criterion(df):
    counts = df['label_int'].value_counts().sort_index().values
    weights = counts.sum() / (len(counts) * counts)
    return FocalLoss(weight=torch.tensor(weights, dtype=torch.float).to(DEVICE), gamma=2.0)

def train():
    train_df, val_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)
    train_ds = CytologyDataset(train_df, IMAGE_DIR, is_train=True)
    val_ds = CytologyDataset(val_df, IMAGE_DIR, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(DEVICE)

    writer = SummaryWriter(LOG_DIR)
    criterion = get_weighted_criterion(train_df)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.05)
    
    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint.")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS,
        last_epoch=start_epoch * len(train_loader) - 1 if start_epoch > 0 else -1
    )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for imgs, labels, _ in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # Validation
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        v_loss, correct, total, scc_c, scc_t = 0.0, 0, 0, 0, 0
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                v_loss += criterion(out, labels).item()
                _, pred = torch.max(out, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                scc_mask = (labels == 4)
                scc_t += scc_mask.sum().item()
                scc_c += (pred[scc_mask] == 4).sum().item()

        avg_val_loss, val_acc = v_loss / len(val_loader), 100 * correct / total
        scc_recall = 100 * scc_c / scc_t if scc_t > 0 else 0
        
        print(f"Epoch {epoch+1}: T-Loss {avg_train_loss:.4f} | V-Loss {avg_val_loss:.4f} | Acc {val_acc:.2f}% | SCC {scc_recall:.2f}%")

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Recall/SCC', scc_recall, epoch)

        ckpt = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_acc': best_acc}
        torch.save(ckpt, CHECKPOINT_PATH)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New High Score: {val_acc:.2f}% (SCC: {scc_recall:.2f}%) - Saved.")

    writer.close()

if __name__ == "__main__":
    train()