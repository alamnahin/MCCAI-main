import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

from dataset import Config as DatasetConfig, RadGenDataset, get_transforms
from encoder_generator import ReportGenerator
from fusion_model import CrossModalFusion, RadGen
from training import CombinedLoss, train_epoch, validate, EarlyStopping, get_optimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "./data/miccai_final_dataset.csv"
    IMG_DIR = "./data/images/"
    TARGET_FOLD = 4
    BATCH_SIZE = 8
    NUM_EPOCHS = 10

    @classmethod
    def set_seed(cls):
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        np.random.seed(cls.SEED)
        import random

        random.seed(cls.SEED)


def main():
    """Main training loop"""
    Config.set_seed()
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Fold: {Config.TARGET_FOLD}")

    # ==================== DATA ====================
    logger.info("\nLoading datasets...")
    try:
        train_ds = RadGenDataset(
            Config.CSV_PATH,
            Config.IMG_DIR,
            fold=Config.TARGET_FOLD,
            split="train",
            transform=get_transforms("train"),
        )

        val_ds = RadGenDataset(
            Config.CSV_PATH,
            Config.IMG_DIR,
            fold=Config.TARGET_FOLD,
            split="val",
            transform=get_transforms("val"),
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=WeightedRandomSampler(train_ds.weights, len(train_ds), True),
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ==================== MODEL ====================
    logger.info("\nInitializing model...")
    model = RadGen(freeze_bert_epochs=3).to(Config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ==================== OPTIMIZER ====================
    optimizer = get_optimizer(model, lr=1e-4)

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * Config.NUM_EPOCHS
    num_warmup_steps = len(train_loader) * 3  # 3 epochs warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # ==================== LOSS ====================
    criterion = CombinedLoss(report_weight=0.3)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # Early stopping
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode="max")

    # ==================== TRAINING LOOP ====================
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_auc": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_auc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        logger.info("-" * 60)

        # FIX: Set epoch for progressive unfreezing
        model.set_epoch(epoch)

        # Log current learning rates
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        logger.info(f"Learning rates: {current_lrs}")

        # Train
        train_loss, cls_loss, rep_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE, scaler, scheduler
        )

        # Validate
        val_metrics, val_probs, val_labels = validate(model, val_loader, Config.DEVICE)

        # Logging
        logger.info(
            f"Train Loss: {train_loss:.4f} (CLS: {cls_loss:.4f}, REP: {rep_loss:.4f}) | "
            f"Train Acc: {train_acc:.4f}"
        )
        logger.info(
            f"Val AUC: {val_metrics['macro_auc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # Per-class metrics
        for cls in ["Normal", "Pneumonia", "TB"]:
            logger.info(
                f"  {cls:10s} - AUC: {val_metrics[f'{cls.lower()}_auc']:.4f}, "
                f"F1: {val_metrics[f'{cls.lower()}_f1']:.4f}"
            )

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_auc"].append(val_metrics["macro_auc"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["macro_f1"])

        # Early stopping check
        is_best = early_stopping(val_metrics["macro_auc"], model)

        if is_best:
            best_auc = val_metrics["macro_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": best_auc,
                    "metrics": val_metrics,
                    "history": history,
                },
                f"best_radgen_fold{Config.TARGET_FOLD}.pth",
            )
            logger.info(f"*** New best model saved! AUC: {best_auc:.4f} ***")

        if early_stopping.early_stop:
            logger.info(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break

    # ==================== FINAL EVALUATION ====================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    # Load best model
    model.load_state_dict(early_stopping.best_model)
    final_metrics, final_probs, final_labels = validate(
        model, val_loader, Config.DEVICE
    )

    logger.info(f"\nBest Macro AUC: {final_metrics['macro_auc']:.4f}")
    logger.info(f"Best Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Best Macro F1: {final_metrics['macro_f1']:.4f}")

    # Print sample generated reports
    logger.info("\nSample Generated Reports:")
    for i in range(min(5, len(final_metrics["generated_reports"]))):
        logger.info(f"  {i + 1}. {final_metrics['generated_reports'][i][:100]}...")

    # Plot training history
    plot_history(history)

    # Plot confusion matrix
    plot_confusion_matrix(final_metrics["confusion_matrix"])

    return model, history, final_metrics


def plot_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    # AUC
    axes[1, 0].plot(history["val_auc"], marker="o", color="green")
    axes[1, 0].set_title("Validation AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim([0, 1])

    # F1
    axes[1, 1].plot(history["val_f1"], marker="s", color="blue")
    axes[1, 1].set_title("Validation F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300)
    plt.show()


def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix"""
    if class_names is None:
        class_names = ["Normal", "Pneumonia", "TB"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


@torch.no_grad()
def inference(model, image_path, transform=None):
    """Single image inference"""
    from PIL import Image

    if transform is None:
        transform = get_transforms("val")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

    # Generate report and classify
    model.eval()
    preds, probs, reports, outputs = model.generate_and_classify(image_tensor)

    class_names = ["Normal", "Pneumonia", "TB"]

    return {
        "report": reports[0],
        "prediction": class_names[preds[0].item()],
        "probabilities": {cls: probs[0, i].item() for i, cls in enumerate(class_names)},
    }


if __name__ == "__main__":
    model, history, metrics = main()

