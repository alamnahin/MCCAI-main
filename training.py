import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)

class Config:
    CLASS_WEIGHTS = torch.tensor([8.0, 1.0, 3.2])
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CombinedLoss(nn.Module):
    """
    Combined loss for joint training:
    - Classification loss (weighted cross-entropy)
    - Report generation loss (from BART)
    """
    def __init__(self, report_weight=0.3):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=Config.CLASS_WEIGHTS.to(Config.DEVICE)
        )
        self.report_weight = report_weight
    
    def forward(self, model_outputs, labels):
        """
        Args:
            model_outputs: dict from RadGen.forward()
            labels: [B] class labels
        Returns:
            total_loss, cls_loss_value, report_loss_value
        """
        # Classification loss
        cls_loss = self.ce_loss(model_outputs['logits'], labels)
        
        # Report generation loss (if available)
        if model_outputs['report_loss'] is not None:
            total_loss = cls_loss + self.report_weight * model_outputs['report_loss']
            return total_loss, cls_loss.item(), model_outputs['report_loss'].item()
        
        return cls_loss, cls_loss.item(), 0.0


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, scheduler=None):
    """
    Train for one epoch with mixed precision support
    """
    model.train()
    total_loss = 0.0
    cls_loss_sum = 0.0
    report_loss_sum = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        # FIX: Don't use clinical_ids from batch during training - model generates them
        # We only use report_ids for BART teacher forcing
        report_ids = batch['report_input_ids'].to(device)
        report_mask = batch['report_attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        # FIX: Use new torch.amp API instead of deprecated torch.cuda.amp
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images, 
                              report_ids=report_ids, 
                              report_mask=report_mask,
                              mode='joint')
                loss, cls_l, rep_l = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, 
                          report_ids=report_ids, 
                          report_mask=report_mask,
                          mode='joint')
            loss, cls_l, rep_l = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        cls_loss_sum += cls_l
        report_loss_sum += rep_l
        
        _, predicted = outputs['logits'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        
        if scheduler is not None:
            scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = cls_loss_sum / len(dataloader)
    avg_report_loss = report_loss_sum / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, avg_cls_loss, avg_report_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, device, class_names=None):
    """
    Validation with comprehensive metrics
    """
    if class_names is None:
        class_names = ['Normal', 'Pneumonia', 'TB']
    
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    all_reports = []  # FIX: Store generated reports
    
    for batch in tqdm(dataloader, desc="Validation"):
        images = batch['image'].to(device)
        labels = batch['label']
        
        # Use generate_and_classify to get reports and predictions
        preds, probs, reports, _ = model.generate_and_classify(images)
        
        all_probs.append(probs.cpu())
        all_labels.append(labels)
        all_preds.append(preds.cpu())
        all_reports.extend(reports)
    
    # Concatenate all batches
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    # Calculate metrics
    y_true_bin = label_binarize(all_labels, classes=[0, 1, 2])
    
    metrics = {
        'accuracy': (all_preds == all_labels).mean(),
        'macro_auc': roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr'),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'generated_reports': all_reports  # FIX: Return reports for analysis
    }
    
    # Per-class metrics
    for i, name in enumerate(class_names):
        try:
            metrics[f'{name.lower()}_auc'] = roc_auc_score(
                y_true_bin[:, i], all_probs[:, i]
            )
        except ValueError:
            metrics[f'{name.lower()}_auc'] = 0.0
            
        metrics[f'{name.lower()}_f1'] = f1_score(
            all_labels == i, all_preds == i, zero_division=0
        )
        
        # Sensitivity / Recall
        tp = metrics['confusion_matrix'][i, i]
        fn = metrics['confusion_matrix'][i, :].sum() - tp
        metrics[f'{name.lower()}_sens'] = tp / (tp + fn + 1e-10)
    
    return metrics, all_probs, all_labels


class EarlyStopping:
    """Early stopping with patience"""
    def __init__(self, patience=10, delta=0.001, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, score, model):
        import copy
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def get_optimizer(model, lr=1e-4):
    """
    Get optimizer with different learning rates for different components.
    Uses parameter IDs to prevent overlap.
    """
    # 1. Group parameters by component
    swin_params = list(model.report_gen.image_encoder.backbone.parameters())
    bart_params = list(model.report_gen.bart.parameters())
    bert_params = list(model.clinical_bert.parameters())
    
    # 2. Collect IDs of parameters already assigned
    handled_ids = set(id(p) for p in swin_params + bart_params + bert_params)
    
    # 3. Everything else (projections, fusion, classifier)
    rest_params = [p for p in model.parameters() if id(p) not in handled_ids]
    
    param_groups = [
        # Pretrained backbones (lower LR)
        {'params': swin_params, 'lr': lr * 0.1, 'weight_decay': 0.01},
        {'params': bart_params, 'lr': lr * 0.2, 'weight_decay': 0.01},
        {'params': bert_params, 'lr': lr * 0.2, 'weight_decay': 0.01},
        # New layers (standard LR)
        {'params': rest_params, 'lr': lr,       'weight_decay': 0.001}
    ]
    
    return torch.optim.AdamW(param_groups)