import torch
import torch.nn as nn
from transformers import SwinModel, BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput

class ImageEncoderWrapper(nn.Module):
    """
    Wrapper to expose Swin Transformer features in the format expected by FusionModel.
    Returns [B, 512] global features.
    """
    def __init__(self, swin_model, fusion_dim=512):
        super().__init__()
        self.backbone = swin_model
        # Swin Base has 1024 hidden units, project to 512 for fusion
        self.proj = nn.Linear(1024, fusion_dim)
    
    def forward(self, x):
        # 1. Forward pass through Swin
        out = self.backbone(x)
        
        # 2. Global Average Pooling on the sequence: [B, 49, 1024] -> [B, 1024]
        features = out.last_hidden_state.mean(dim=1)
        
        # 3. Project to fusion dimension
        return self.proj(features)  # [B, 512]

class ReportGenerator(nn.Module):
    """
    Generates radiology reports using Swin Transformer + BART.
    """
    def __init__(self):
        super().__init__()
        # 1. Image Encoder (Swin Base)
        self.swin = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
        
        # 2. Report Decoder (BART Base)
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
        # 3. Adapter: Project Swin features (1024) to BART embedding size (768)
        # FIX: Added LayerNorm and GELU per paper description
        self.bart_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )
        
        # 4. Expose image encoder for the Fusion module (expects [B, 512])
        self.image_encoder = ImageEncoderWrapper(self.swin, fusion_dim=512)
        
    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Training step: Returns BART loss
        """
        # Get image patch embeddings [B, 49, 1024]
        swin_out = self.swin(images)
        visual_embeds = self.bart_proj(swin_out.last_hidden_state)  # [B, 49, 768]
        
        # BART Forward pass
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=(visual_embeds,),  # FIX: Proper encoder_outputs format
            labels=labels
        )
        return outputs

    def generate_report(self, images, max_length=64, num_beams=4):
        """
        Inference step: Generates text token IDs
        """
        # Get image features
        swin_out = self.swin(images)
        visual_embeds = self.bart_proj(swin_out.last_hidden_state)
        
        # FIX: Use BaseModelOutput for proper generation
        encoder_outputs = BaseModelOutput(last_hidden_state=visual_embeds)
        
        generated_ids = self.bart.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        return generated_ids