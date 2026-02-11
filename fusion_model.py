import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from encoder_generator import ReportGenerator

class Config:
    CLINICAL_BERT = 'emilyalsentzer/Bio_ClinicalBERT'
    MAX_LEN = 128

class CrossModalFusion(nn.Module):
    """
    Cross-Modal Attention Fusion with Gating Mechanism
    Fuses image features (512-dim) with text features (768-dim from ClinicalBERT)
    """
    def __init__(self, img_dim=512, text_dim=768, hidden_dim=512, num_heads=8):
        super().__init__()
        
        # Project text to same dimension as image
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention: Image attends to text
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Self-attention for refinement
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, img_features, text_features, text_mask):
        """
        Args:
            img_features: [B, 512] from SwinEncoder
            text_features: [B, seq_len, 768] from ClinicalBERT
            text_mask: [B, seq_len] attention mask (1 for valid, 0 for pad)
        Returns:
            fused_features: [B, 512]
        """
        # Project text features
        text_proj = self.text_proj(text_features)  # [B, seq_len, 512]
        
        # Add sequence dimension to image features
        img_seq = img_features.unsqueeze(1)  # [B, 1, 512]
        
        # Cross-attention: Image queries attend to text keys/values
        key_mask = ~text_mask.bool()  # True where padding
        cross_out, attn_weights = self.cross_attn(
            img_seq, text_proj, text_proj,
            key_padding_mask=key_mask
        )
        
        # Residual connection and normalization
        cross_out = self.norm1(img_seq + cross_out)  # [B, 1, 512]
        
        # Self-attention for refinement
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out)
        self_out = self.norm2(cross_out + self_out)
        
        # Feed-forward
        ffn_out = self_out + self.ffn(self_out)
        
        # Squeeze sequence dimension
        fused = ffn_out.squeeze(1)  # [B, 512]
        
        # Apply gating mechanism (if gate is available)
        if self.gate is not None:
            text_pooled = text_features.mean(dim=1)  # [B, 768]
            text_proj_gate = self.text_proj(text_pooled)  # [B, 512]
            
            gate_input = torch.cat([img_features, text_proj_gate], dim=-1)  # [B, 1024]
            g = self.gate(gate_input)  # [B, 512]
            
            return g * img_features + (1 - g) * fused
        
        return fused


class RadGen(nn.Module):
    """
    Complete RadGen Model:
    1. Generate reports from images (ReportGenerator)
    2. Encode reports with ClinicalBERT
    3. Fuse image + text with cross-attention
    4. Classify with MLP
    """
    def __init__(self, freeze_bert_epochs=3):
        super().__init__()
        
        # 1. Report Generator (Swin + BART)
        self.report_gen = ReportGenerator()
        
        # 2. Clinical BERT for text encoding
        self.clinical_bert = AutoModel.from_pretrained(Config.CLINICAL_BERT)
        
        # Freeze BERT initially for stable training
        self.freeze_bert_epochs = freeze_bert_epochs
        self.current_epoch = 0
        self._freeze_clinical_bert(True)
        
        # 3. Cross-modal fusion
        self.fusion = CrossModalFusion(img_dim=512, text_dim=768, hidden_dim=512)
        
        # 4. Classification head with heavy dropout (regularization)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 classes: Normal, Pneumonia, TB
        )
        
        # Clinical tokenizer for generated reports
        self.clinical_tokenizer = AutoTokenizer.from_pretrained(Config.CLINICAL_BERT)
    
    def _freeze_clinical_bert(self, freeze):
        """Freeze or unfreeze Clinical BERT parameters"""
        for param in self.clinical_bert.parameters():
            param.requires_grad = not freeze
    
    def set_epoch(self, epoch):
        """Call at the start of each epoch to manage freezing"""
        self.current_epoch = epoch
        if epoch == self.freeze_bert_epochs:
            print(f"Epoch {epoch}: Unfreezing Clinical BERT")
            self._freeze_clinical_bert(False)
    
    def forward(self, images, clinical_ids=None, clinical_mask=None,
                report_ids=None, report_mask=None, labels=None, mode='joint'):
        """
        Args:
            images: [B, 3, 224, 224]
            clinical_ids: [B, seq_len] - tokenized reports (ClinicalBERT)
            clinical_mask: [B, seq_len]
            report_ids: [B, seq_len] - tokenized reports (BART)
            report_mask: [B, seq_len]
            labels: [B] - class labels
            mode: 'joint' (train both), 'generate' (only report gen), 
                  'classify' (only classification)
        
        Returns:
            dict with logits, features, and report_loss
        """
        B = images.size(0)
        
        # 1. Get image features
        img_features = self.report_gen.image_encoder(images)
        
        # 2. Report generation (if training or no clinical_ids provided)
        report_loss = None
        if report_ids is not None and mode in ['joint', 'generate']:
            report_output = self.report_gen(images, report_ids, report_mask, labels=report_ids)
            report_loss = report_output.loss
        
        # 3. CRITICAL FIX: Text encoding strategy
        if self.training and mode == 'joint':
            # TRAINING: Always generate reports to prevent data leakage
            # No gradient through generation (detached)
            with torch.no_grad():
                gen_ids = self.report_gen.generate_report(images)
                gen_texts = [self.clinical_tokenizer.decode(g, skip_special_tokens=True) 
                            for g in gen_ids]
                
                # Re-tokenize with ClinicalBERT
                enc = self.clinical_tokenizer(
                    gen_texts, max_length=Config.MAX_LEN,
                    padding='max_length', truncation=True, return_tensors='pt'
                )
                clinical_ids = enc['input_ids'].to(images.device)
                clinical_mask = enc['attention_mask'].to(images.device)
            
            # Enable gradients for ClinicalBERT encoding
            # FIX: Use self.freeze_bert_epochs instead of hardcoded 3
            with torch.set_grad_enabled(self.current_epoch >= self.freeze_bert_epochs):
                text_out = self.clinical_bert(
                    input_ids=clinical_ids,
                    attention_mask=clinical_mask
                )
                text_features = text_out.last_hidden_state
        else:
            # VALIDATION/INFERENCE: Use provided clinical_ids or generate
            if clinical_ids is None:
                with torch.no_grad():
                    gen_ids = self.report_gen.generate_report(images)
                    gen_texts = [self.clinical_tokenizer.decode(g, skip_special_tokens=True) 
                                for g in gen_ids]
                    enc = self.clinical_tokenizer(
                        gen_texts, max_length=Config.MAX_LEN,
                        padding='max_length', truncation=True, return_tensors='pt'
                    )
                    clinical_ids = enc['input_ids'].to(images.device)
                    clinical_mask = enc['attention_mask'].to(images.device)
            
            text_out = self.clinical_bert(
                input_ids=clinical_ids,
                attention_mask=clinical_mask
            )
            text_features = text_out.last_hidden_state
        
        # 4. Cross-modal fusion
        fused_features = self.fusion(img_features, text_features, clinical_mask)
        
        # 5. Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'fused_features': fused_features,
            'img_features': img_features,
            'report_loss': report_loss
        }
    
    def generate_and_classify(self, images):
        """End-to-end inference: Image -> Report -> Classification"""
        self.eval()
        with torch.no_grad():
            # Generate reports
            gen_ids = self.report_gen.generate_report(images)
            gen_texts = [self.clinical_tokenizer.decode(g, skip_special_tokens=True) 
                        for g in gen_ids]
            
            # Get classification
            outputs = self.forward(images)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs, gen_texts, outputs


# ==================== TEST ====================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing CrossModalFusion...")
    fusion = CrossModalFusion().to(device)
    img_feat = torch.randn(2, 512).to(device)
    text_feat = torch.randn(2, 50, 768).to(device)
    text_mask = torch.ones(2, 50).to(device)
    fused = fusion(img_feat, text_feat, text_mask)
    print(f"Fused features shape: {fused.shape}")  # [2, 512]
    
    print("\nTesting complete RadGen model...")
    model = RadGen().to(device)
    
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    dummy_report_ids = torch.randint(0, 1000, (2, 20)).to(device)
    dummy_report_mask = torch.ones(2, 20).to(device)
    
    # Test training mode (should generate reports internally)
    model.train()
    outputs = model(dummy_img, report_ids=dummy_report_ids, 
                   report_mask=dummy_report_mask, mode='joint')
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Report loss: {outputs['report_loss']}")
    
    # Test inference mode
    model.eval()
    preds, probs, reports, _ = model.generate_and_classify(dummy_img)
    print(f"Inference predictions: {preds}")
    print(f"Generated reports: {reports}")
    
    print("\nAll tests passed!")