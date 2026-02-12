import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoTokenizer, BartTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================


class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "./data/miccai_final_dataset.csv"
    IMG_DIR = "./data/images/"
    TARGET_FOLD = 4

    # Model components
    IMAGE_ENCODER = "microsoft/swin-base-patch4-window7-224"
    REPORT_DECODER = "facebook/bart-base"
    CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"

    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LR = 1e-4
    WARMUP_EPOCHS = 3
    MAX_LEN = 128

    # Class weights (inverse frequency: Normal=8x, Pneumonia=1x, TB=3.2x)
    CLASS_WEIGHTS = torch.tensor([8.0, 1.0, 3.2])


def set_seed():
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed_all(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# ==================== DATASET ====================


class RadGenDataset(Dataset):
    """
    Dataset for RadGen: Returns image, label, and tokenized reports
    """

    def __init__(self, csv_file, root_dir, fold=0, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # Load and validate CSV
        df = pd.read_csv(csv_file)

        # FIX: Validate required columns exist
        required_cols = ["Image", "Category", "fold", "Impression"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        df["Image"] = df["Image"].str.strip()
        df = df.dropna(subset=["Image", "Category"])

        # FIX: Validate fold values
        if not df["fold"].between(0, 4).all():
            raise ValueError("Fold values must be between 0 and 4")

        # Split by fold
        if split == "train":
            self.data = df[df["fold"] != fold].reset_index(drop=True)
        else:
            self.data = df[df["fold"] == fold].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(f"No data found for fold {fold}, split {split}")

        self.class_to_idx = {"Normal": 0, "Pneumonia": 1, "TB": 2}

        # Validate all classes present
        unique_classes = self.data["Category"].unique()
        invalid_classes = set(unique_classes) - set(self.class_to_idx.keys())
        if invalid_classes:
            raise ValueError(f"Invalid categories in data: {invalid_classes}")

        # Tokenizers
        self.report_tokenizer = BartTokenizer.from_pretrained(Config.REPORT_DECODER)
        self.clinical_tokenizer = AutoTokenizer.from_pretrained(Config.CLINICAL_BERT)

        # Compute sampling weights for balanced batches
        counts = self.data["Category"].value_counts()
        weights = {cls: len(self.data) / (3 * counts[cls]) for cls in counts.index}
        self.weights = [weights[cat] for cat in self.data["Category"]]

        logger.info(f"[{split}] Loaded {len(self.data)} samples")
        logger.info(f"Class distribution: {dict(counts)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image with proper error handling
        img_path = self.root_dir / row["Image"].strip()
        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # FIX: Log error instead of silent failure
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a blank image but log the error
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["Category"]]

        # Get report text (ground truth for training BART)
        report = (
            str(row.get("Impression", ""))
            if pd.notna(row.get("Impression"))
            else "no finding"
        )
        report = " ".join(report.split())  # Clean whitespace

        # Tokenize for BART (report generation - teacher forcing)
        r_enc = self.report_tokenizer(
            report,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize for ClinicalBERT (classification - will be replaced by generated reports in training)
        c_enc = self.clinical_tokenizer(
            report,
            max_length=Config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "report_input_ids": r_enc["input_ids"].squeeze(0),
            "report_attention_mask": r_enc["attention_mask"].squeeze(0),
            "clinical_input_ids": c_enc["input_ids"].squeeze(0),
            "clinical_attention_mask": c_enc["attention_mask"].squeeze(0),
            "gt_report": report,  # Keep for reference
            "category": row["Category"],
            "image_name": row["Image"],  # FIX: Keep for debugging
        }


# ==================== TRANSFORMS ====================


def get_transforms(split="train"):
    """Get data augmentation transforms"""
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


if __name__ == "__main__":
    # Test dataset
    try:
        train_ds = RadGenDataset(
            Config.CSV_PATH,
            Config.IMG_DIR,
            fold=Config.TARGET_FOLD,
            split="train",
            transform=get_transforms("train"),
        )
        print(f"\nSample batch keys: {train_ds[0].keys()}")
        print(f"Image shape: {train_ds[0]['image'].shape}")
        print(f"Report: {train_ds[0]['gt_report'][:100]}...")
        print(f"Image name: {train_ds[0]['image_name']}")
    except Exception as e:
        logger.error(f"Dataset test failed: {e}")

