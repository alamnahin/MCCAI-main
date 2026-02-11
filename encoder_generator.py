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
        self.proj = nn.Linear(1024, fusion_dim)
    
    def forward(self, x):
        out = self.backbone(x)
        features = out.last_hidden_state.mean(dim=1)
        return self.proj(features)

class ReportGenerator(nn.Module):
    """
    Generates radiology reports using Swin Transformer + BART.
    """
    def __init__(self):
        super().__init__()
        self.swin = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
        
        # BEST FIX: Use eager attention to avoid SDPA dimension constraints
        self.bart = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base',
            attn_implementation="eager"  # Disables SDPA, uses standard attention
        )
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
        # Adapter: Project Swin features to BART embedding size
        self.bart_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )
        
        self.image_encoder = ImageEncoderWrapper(self.swin, fusion_dim=512)
        
    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Training step: Returns BART loss
        """
        # Get image features [B, 49, 1024]
        swin_out = self.swin(images)
        visual_embeds = self.bart_proj(swin_out.last_hidden_state)  # [B, 49, 768]
        
        # Create encoder attention mask
        encoder_attention_mask = torch.ones(
            visual_embeds.size()[:2],
            dtype=torch.long,
            device=visual_embeds.device
        )
        
        # Process through BART's encoder
        encoder_outputs = self.bart.model.encoder(
            inputs_embeds=visual_embeds,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )
        
        # Call BART with processed encoder outputs
        # FIX: attention_mask = encoder mask (length 49), decoder_attention_mask = text mask (length 64)
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            decoder_attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return outputs

    def generate_report(self, images, max_length=64, num_beams=4, **kwargs):
        """
        Inference step: Generates text token IDs
        """
        swin_out = self.swin(images)
        visual_embeds = self.bart_proj(swin_out.last_hidden_state)
        
        # Create encoder attention mask
        encoder_attention_mask = torch.ones(
            visual_embeds.size()[:2],
            dtype=torch.long,
            device=visual_embeds.device
        )
        
        # Process through encoder
        encoder_outputs = self.bart.model.encoder(
            inputs_embeds=visual_embeds,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )
        
        # Generate
        # FIX: Pass attention_mask (encoder mask) instead of non-standard encoder_attention_mask
        generated_ids = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            decoder_start_token_id=self.bart.config.decoder_start_token_id,
            eos_token_id=self.bart.config.eos_token_id,
            pad_token_id=self.bart.config.pad_token_id,
            **kwargs
        )
        return generated_ids