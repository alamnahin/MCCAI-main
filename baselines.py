import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

from dataset import Config as DatasetConfig, RadGenDataset, get_transforms
from training import validate
import logging

logger = logging.getLogger(__name__)

class DenseNet121Baseline(nn.Module):
    """DenseNet121 image-only baseline"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.densenet121(pretrained=True)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, images):
        return {'logits': self.backbone(images)}

class SwinBaseline(nn.Module):
    """Swin Transformer image-only baseline"""
    def __init__(self, num_classes=3):
        super().__init__()
        from transformers import SwinModel
        self.swin = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images):
        features = self.swin(images).last_hidden_state.mean(dim=1)
        return {'logits': self.classifier(features)}

class ReportOnlyBaseline(nn.Module):
    """ClinicalBERT text-only baseline"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, clinical_ids, clinical_mask):
        outputs = self.bert(input_ids=clinical_ids, attention_mask=clinical_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return {'logits': self.classifier(pooled)}

class EarlyFusionBaseline(nn.Module):
    """Concatenate image features + text embeddings"""
    def __init__(self, num_classes=3):
        super().__init__()
        from transformers import SwinModel, AutoModel
        
        self.swin = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
        self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, clinical_ids, clinical_mask):
        img_feat = self.swin(images).last_hidden_state.mean(dim=1)
        text_feat = self.bert(input_ids=clinical_ids, attention_mask=clinical_mask).last_hidden_state.mean(dim=1)
        
        fused = self.fusion(torch.cat([img_feat, text_feat], dim=-1))
        return {'logits': self.classifier(fused)}

class LateFusionBaseline(nn.Module):
    """Average predictions from image and text classifiers"""
    def __init__(self, num_classes=3):
        super().__init__()
        from transformers import SwinModel, AutoModel
        
        self.swin = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
        self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        self.img_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.text_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, clinical_ids, clinical_mask):
        img_feat = self.swin(images).last_hidden_state.mean(dim=1)
        text_feat = self.bert(input_ids=clinical_ids, attention_mask=clinical_mask).last_hidden_state.mean(dim=1)
        
        img_logits = self.img_classifier(img_feat)
        text_logits = self.text_classifier(text_feat)
        
        # Average logits
        avg_logits = (img_logits + text_logits) / 2
        return {'logits': avg_logits}


def train_baseline(model, train_loader, val_loader, device, num_epochs=30, lr=1e-4, model_name="baseline"):
    """Train any baseline model"""
    from torch.utils.data import DataLoader
    from training import EarlyStopping, get_optimizer as get_base_optimizer
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([8.0, 1.0, 3.2]).to(device))
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode='max')
    
    best_metrics = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Handle different model inputs
            if isinstance(model, DenseNet121Baseline) or isinstance(model, SwinBaseline):
                outputs = model(batch['image'].to(device))
                labels = batch['label'].to(device)
            elif isinstance(model, ReportOnlyBaseline):
                outputs = model(batch['clinical_input_ids'].to(device), 
                              batch['clinical_attention_mask'].to(device))
                labels = batch['label'].to(device)
            else:
                outputs = model(batch['image'].to(device),
                              batch['clinical_input_ids'].to(device),
                              batch['clinical_attention_mask'].to(device))
                labels = batch['label'].to(device)
            
            loss = criterion(outputs['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        val_metrics = validate_baseline(model, val_loader, device)
        logger.info(f"{model_name} Epoch {epoch+1}: Val AUC={val_metrics['macro_auc']:.4f}, "
                   f"Acc={val_metrics['accuracy']:.4f}")
        
        if early_stopping(val_metrics['macro_auc'], model):
            best_metrics = val_metrics
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        
        if early_stopping.early_stop:
            break
    
    return best_metrics


@torch.no_grad()
def validate_baseline(model, dataloader, device):
    """Validate baseline models"""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    for batch in tqdm(dataloader, desc="Validation"):
        if isinstance(model, DenseNet121Baseline) or isinstance(model, SwinBaseline):
            outputs = model(batch['image'].to(device))
        elif isinstance(model, ReportOnlyBaseline):
            outputs = model(batch['clinical_input_ids'].to(device),
                          batch['clinical_attention_mask'].to(device))
        else:
            outputs = model(batch['image'].to(device),
                          batch['clinical_input_ids'].to(device),
                          batch['clinical_attention_mask'].to(device))
        
        probs = F.softmax(outputs['logits'], dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_probs.append(probs.cpu())
        all_labels.append(batch['label'])
        all_preds.append(preds.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    y_true_bin = label_binarize(all_labels, classes=[0, 1, 2])
    
    metrics = {
        'accuracy': (all_preds == all_labels).mean(),
        'macro_auc': roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr'),
        'macro_f1': f1_score(all_labels, all_preds, average='macro')
    }
    
    # Per-class metrics
    class_names = ['Normal', 'Pneumonia', 'TB']
    for i, name in enumerate(class_names):
        try:
            metrics[f'{name.lower()}_auc'] = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        except ValueError:
            metrics[f'{name.lower()}_auc'] = 0.0
        metrics[f'{name.lower()}_f1'] = f1_score(all_labels == i, all_preds == i, zero_division=0)
    
    return metrics


def run_all_baselines(fold=4, batch_size=8):
    """Run all baseline experiments"""
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_ds = RadGenDataset(DatasetConfig.CSV_PATH, DatasetConfig.IMG_DIR,
                            fold=fold, split='train', transform=get_transforms('train'))
    val_ds = RadGenDataset(DatasetConfig.CSV_PATH, DatasetConfig.IMG_DIR,
                          fold=fold, split='val', transform=get_transforms('val'))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    results = {}
    
    # 1. DenseNet121
    logger.info("\n" + "="*60)
    logger.info("Training DenseNet121 Baseline")
    logger.info("="*60)
    model = DenseNet121Baseline().to(device)
    results['densenet121'] = train_baseline(model, train_loader, val_loader, device, 
                                           model_name='densenet121')
    
    # 2. Swin-B
    logger.info("\n" + "="*60)
    logger.info("Training Swin-B Baseline")
    logger.info("="*60)
    model = SwinBaseline().to(device)
    results['swin_b'] = train_baseline(model, train_loader, val_loader, device,
                                      model_name='swin_b')
    
    # 3. Report-only (ClinicalBERT)
    logger.info("\n" + "="*60)
    logger.info("Training Report-Only Baseline")
    logger.info("="*60)
    model = ReportOnlyBaseline().to(device)
    results['report_only'] = train_baseline(model, train_loader, val_loader, device,
                                           model_name='report_only')
    
    # 4. Early Fusion
    logger.info("\n" + "="*60)
    logger.info("Training Early Fusion Baseline")
    logger.info("="*60)
    model = EarlyFusionBaseline().to(device)
    results['early_fusion'] = train_baseline(model, train_loader, val_loader, device,
                                            model_name='early_fusion')
    
    # 5. Late Fusion
    logger.info("\n" + "="*60)
    logger.info("Training Late Fusion Baseline")
    logger.info("="*60)
    model = LateFusionBaseline().to(device)
    results['late_fusion'] = train_baseline(model, train_loader, val_loader, device,
                                           model_name='late_fusion')
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    results = run_all_baselines()
    
    # Print summary table
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Macro AUC':>10} {'Accuracy':>10} {'Normal AUC':>12} {'Pneumonia AUC':>14} {'TB AUC':>10}")
    print("-"*80)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['macro_auc']:>10.4f} {metrics['accuracy']:>10.4f} "
              f"{metrics.get('normal_auc', 0):>12.4f} {metrics.get('pneumonia_auc', 0):>14.4f} "
              f"{metrics.get('tb_auc', 0):>10.4f}")