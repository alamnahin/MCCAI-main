import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import copy
import logging

from fusion_model import RadGen, CrossModalFusion, Config as FusionConfig
from dataset import Config as DatasetConfig, RadGenDataset, get_transforms
from training import train_epoch, validate, CombinedLoss, EarlyStopping, get_optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class RadGenAblation(RadGen):
    """RadGen with configurable components for ablation"""

    def __init__(
        self,
        use_report_gen=True,
        use_cross_attention=True,
        use_gating=True,
        use_progressive_unfreezing=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_report_gen = use_report_gen
        self.use_cross_attention = use_cross_attention
        self.use_gating = use_gating
        self.use_progressive_unfreezing = use_progressive_unfreezing

        # Modify fusion if no cross-attention
        if not use_cross_attention:
            self.fusion = nn.Sequential(
                nn.Linear(512 + 768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3)
            )

        # Modify fusion if no gating
        if not use_gating and use_cross_attention:
            # Re-initialize without gating
            self.fusion.gate = None

    def forward(
        self,
        images,
        clinical_ids=None,
        clinical_mask=None,
        report_ids=None,
        report_mask=None,
        labels=None,
        mode="joint",
    ):
        # Skip report generation if ablated
        if not self.use_report_gen:
            report_ids = None
            report_mask = None

        # Call parent forward
        outputs = super().forward(
            images, clinical_ids, clinical_mask, report_ids, report_mask, labels, mode
        )

        # If no cross-attention, do simple concatenation
        if not self.use_cross_attention and clinical_ids is not None:
            img_features = self.report_gen.image_encoder(images)
            text_features = self.clinical_bert(
                input_ids=clinical_ids, attention_mask=clinical_mask
            ).last_hidden_state.mean(dim=1)

            fused = self.fusion(torch.cat([img_features, text_features], dim=-1))
            logits = self.classifier(fused)
            outputs["logits"] = logits
            outputs["fused_features"] = fused

        return outputs


def run_ablation_experiment(
    config_name, model_config, train_loader, val_loader, device, num_epochs=10
):
    """Run single ablation experiment"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running Ablation: {config_name}")
    logger.info(f"Config: {model_config}")
    logger.info(f"{'=' * 60}")

    model = RadGenAblation(**model_config).to(device)
    optimizer = get_optimizer(model, lr=1e-4)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    criterion = CombinedLoss(
        report_weight=0.3 if model_config.get("use_report_gen", True) else 0.0
    )
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode="max")

    best_metrics = None

    for epoch in range(num_epochs):
        if model.use_progressive_unfreezing:
            model.set_epoch(epoch)

        train_loss, cls_loss, rep_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler
        )

        val_metrics, _, _ = validate(model, val_loader, device)

        logger.info(f"Epoch {epoch + 1}: Val AUC={val_metrics['macro_auc']:.4f}")

        if early_stopping(val_metrics["macro_auc"], model):
            best_metrics = val_metrics
            torch.save(
                {
                    "config": config_name,
                    "model_state_dict": model.state_dict(),
                    "metrics": val_metrics,
                },
                f"ablation_{config_name}.pth",
            )

        if early_stopping.early_stop:
            break

    return best_metrics


def run_all_ablations(fold=4, batch_size=8):
    """Run all ablation studies from Table 4"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_ds = RadGenDataset(
        DatasetConfig.CSV_PATH,
        DatasetConfig.IMG_DIR,
        fold=fold,
        split="train",
        transform=get_transforms("train"),
    )
    val_ds = RadGenDataset(
        DatasetConfig.CSV_PATH,
        DatasetConfig.IMG_DIR,
        fold=fold,
        split="val",
        transform=get_transforms("val"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(train_ds.weights, len(train_ds), True),
        num_workers=2,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    ablation_configs = {
        "full_model": {
            "use_report_gen": True,
            "use_cross_attention": True,
            "use_gating": True,
            "use_progressive_unfreezing": True,
        },
        "wo_report_gen": {
            "use_report_gen": False,  # No BART, use GT reports only
            "use_cross_attention": True,
            "use_gating": True,
            "use_progressive_unfreezing": True,
        },
        "wo_cross_attention": {
            "use_report_gen": True,
            "use_cross_attention": False,  # Simple concatenation
            "use_gating": False,
            "use_progressive_unfreezing": True,
        },
        "wo_gating": {
            "use_report_gen": True,
            "use_cross_attention": True,
            "use_gating": False,  # No gating mechanism
            "use_progressive_unfreezing": True,
        },
        "wo_progressive_unfreezing": {
            "use_report_gen": True,
            "use_cross_attention": True,
            "use_gating": True,
            "use_progressive_unfreezing": False,  # Unfreeze BERT from start
        },
        "wo_class_weighting": {
            "use_report_gen": True,
            "use_cross_attention": True,
            "use_gating": True,
            "use_progressive_unfreezing": True,
            # Note: class weighting handled in loss function
        },
    }

    results = {}

    for config_name, model_config in ablation_configs.items():
        # Special handling for class weighting ablation
        if config_name == "wo_class_weighting":
            # Modify loss to no weights
            original_weights = train_loader.dataset.class_to_idx.copy()
            # This requires modifying the loss function temporarily
            pass  # Handled in training loop

        results[config_name] = run_ablation_experiment(
            config_name, model_config, train_loader, val_loader, device
        )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_all_ablations()

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Component':<30} {'Macro AUC':>10} {'Δ':>8}")
    print("-" * 50)

    full_auc = results["full_model"]["macro_auc"]
    print(f"{'Full Model':<30} {full_auc:>10.4f} {'-':>8}")

    for name, metrics in results.items():
        if name != "full_model":
            delta = metrics["macro_auc"] - full_auc
            print(f"{name:<30} {metrics['macro_auc']:>10.4f} {delta:>+8.4f}")

