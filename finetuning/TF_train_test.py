"""
Fine-tunes a pretrained model on binary TF binding site prediction for ONE TF sample.

Training / validation / testing strategy:
  - the dev set is evaluated every epoch,
  - the best checkpoint (highest dev accuracy) is restored at the end of training,
  - training stops early after 3 evaluations without improvement,
  - per-region predictions are saved to CSV for the dev and test sets.

Input is unchanged: --tf_dir containing train.npz, dev.npz and test.npz.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments as HfTrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
)


# ============================================================
# Dataclass arguments
# ============================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/opt-125m",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    factor: float = field(
        default=1.0,
        metadata={"help": "RoPE scaling factor"}
    )
    # Unused; kept so existing command lines still parse
    pretrained_encoder: Optional[str] = field(default=None)
    freeze_encoder: bool = field(default=False)


@dataclass
class DataArguments:
    tf_dir: str = field(
        default=None,
        metadata={"help": "Directory containing train.npz dev.npz test.npz for ONE TF"}
    )


@dataclass
class MyTrainingArguments(HfTrainingArguments):
    output_dir: str = field(default="out")
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=32)
    auto_find_batch_size: bool = field(default=True)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: int = field(default=10)
    seed: int = field(default=42)
    save_strategy: str = field(default="epoch")
    eval_strategy: str = field(default="epoch")
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=50)
    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"])
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="accuracy")
    greater_is_better: bool = field(default=True)


# ============================================================
# Dataset for TF classification
# ============================================================
def format_coord(c):
    """Format a [chr, start, end] coordinate into chr:start-end."""
    return f"{c[0]}:{c[1]}-{c[2]}" if isinstance(c, (list, np.ndarray)) else str(c)


class TFPeakDataset(Dataset):
    """
    Loads one .npz containing:
      X:      (N, L, 10) probability matrices
      y:      (N,) labels
      coords: (N, ...) optional region coordinates; if absent, the row index is used
    """

    def __init__(self, npz_path):
        arr = np.load(npz_path, allow_pickle=True)
        self.seq = arr["X"].astype(np.float32)     # (N, L, 10)
        self.labels = arr["y"].astype(np.int64)    # (N,)

        if "coords" in arr.files:
            self.coords = [format_coord(c) for c in arr["coords"]]
        else:
            self.coords = [str(i) for i in range(len(self.labels))]

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.seq[idx])           # (L,10)
        attn = (x.sum(dim=-1) > 0).float()            # (L,)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": x, "attention_mask": attn, "labels": y}


# ============================================================
# Collate — pad if needed
# ============================================================
def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    xs = [b["input_ids"] for b in batch]   # list of (L,10)
    ms = [b["attention_mask"] for b in batch]
    ys = torch.stack([b["labels"] for b in batch])  # (B,)

    Ls = [x.shape[0] for x in xs]
    maxL = max(Ls)

    padded = []
    masks = []

    for x, m in zip(xs, ms):
        if x.shape[0] < maxL:
            pad = maxL - x.shape[0]
            padded.append(torch.cat([x, torch.zeros(pad, x.shape[1])], dim=0))
            masks.append(torch.cat([m, torch.zeros(pad)], dim=0))
        else:
            padded.append(x)
            masks.append(m)

    return {
        "input_ids": torch.stack(padded, dim=0),     # (B,L,10)
        "attention_mask": torch.stack(masks, dim=0), # (B,L)
        "labels": ys
    }


# ============================================================
# Metrics
# ============================================================
def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]  # P(class=1)
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(labels, probs)
    except ValueError:
        auprc = float("nan")

    return {
        "accuracy": acc,
        "f1": f1,
        "matthews_correlation": mcc,
        "auroc": auroc,
        "auprc": auprc,
    }


# ============================================================
# Helper function to save predictions
# ============================================================
def save_predictions_to_csv(trainer, dataset, split_name, output_dir):
    """Runs prediction on a dataset and saves coordinates, observed labels and predictions to a CSV."""
    print(f"\n[INFO] Running predictions on {split_name} set...")

    pred_output = trainer.predict(dataset, metric_key_prefix=split_name)

    probs = torch.softmax(torch.tensor(pred_output.predictions), dim=-1).numpy()[:, 1]
    preds = np.argmax(pred_output.predictions, axis=-1)
    obs = np.squeeze(pred_output.label_ids)

    df = pd.DataFrame({
        'coordinate': dataset.coords,
        'observed_label': obs,
        'predicted_label': preds,
        'probability_peak': probs
    })

    out_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved {split_name} predictions to: {out_path}")

    return pred_output.metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # paths
    train_p = os.path.join(data_args.tf_dir, "train.npz")
    dev_p   = os.path.join(data_args.tf_dir, "dev.npz")
    test_p  = os.path.join(data_args.tf_dir, "test.npz")

    for p in (train_p, dev_p, test_p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    train_ds = TFPeakDataset(train_p)
    dev_ds   = TFPeakDataset(dev_p)
    test_ds  = TFPeakDataset(test_p)

    print(f"[INFO] Loaded: {len(train_ds)} train; {len(dev_ds)} dev; {len(test_ds)} test.")

    print(f"[INFO] Loading model: {model_args.model_name_or_path}...")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        rope_scaling={'type': 'dynamic', 'factor': model_args.factor},
        num_labels=2,
        trust_remote_code=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_classification,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\n[INFO] Starting Training...")
    trainer.train()

    # The best-on-dev checkpoint has been restored at this point
    os.makedirs(training_args.output_dir, exist_ok=True)
    dev_metrics = save_predictions_to_csv(trainer, dev_ds, "val", training_args.output_dir)
    test_metrics = save_predictions_to_csv(trainer, test_ds, "test", training_args.output_dir)

    print("\n[INFO] Best-checkpoint Validation Metrics:")
    print(dev_metrics)
    print("\n[INFO] Final Test Metrics:")
    print(test_metrics)

    with open(os.path.join(training_args.output_dir, "test_eval_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    main()
