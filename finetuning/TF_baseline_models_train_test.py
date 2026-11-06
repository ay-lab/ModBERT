"""
Trains from-scratch baseline models (CNN, RNN, LSTM, GRU) on binary TF binding
site prediction for ONE TF sample.

Training / validation / testing strategy mirrors:
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
from torch import nn
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
    model_type: str = field(
        default="cnn",
        metadata={"help": "Model type: 'cnn', 'lstm', 'rnn', 'gru'"}
    )
    input_channels: int = field(
        default=10,
        metadata={"help": "Number of channels in the input matrices"}
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
    output_dir: str = field(default="out_ab_initio_classification")
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=64)
    learning_rate: float = field(default=1e-3)
    num_train_epochs: int = field(default=30)
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
        return {"input_probs": x, "attention_mask": attn, "labels": y}


# ============================================================
# Collate — pad if needed
# ============================================================
def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    xs = [b["input_probs"] for b in batch]   # list of (L,10)
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
        "input_probs": torch.stack(padded, dim=0),   # (B,L,10)
        "attention_mask": torch.stack(masks, dim=0), # (B,L)
        "labels": ys
    }


# ============================================================
# Model implementations
# ============================================================
class RNNForBinaryClassification(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=2, rnn_type="rnn"):
        super().__init__()
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, input_probs=None, attention_mask=None, labels=None):
        h, _ = self.rnn(input_probs)
        if attention_mask is not None:
            h = h.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
        pooled, _ = torch.max(h, dim=1)
        logits = self.classifier(pooled)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return transformers.modeling_outputs.SequenceClassifierOutput(loss=loss, logits=logits)


class CNNForBinaryClassification(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input_probs=None, attention_mask=None, labels=None):
        x = input_probs.transpose(1, 2)
        h = self.conv(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            h = h.masked_fill(mask == 0, -1e4)
        pooled, _ = torch.max(h, dim=2)
        logits = self.classifier(pooled)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return transformers.modeling_outputs.SequenceClassifierOutput(loss=loss, logits=logits)


# ============================================================
# Model factory
# ============================================================
def get_model(model_type: str, input_channels: int, seq_len: int):
    model_type = model_type.lower()
    if model_type == "cnn":
        return CNNForBinaryClassification(input_dim=input_channels)
    elif model_type in ["lstm", "rnn", "gru"]:
        return RNNForBinaryClassification(input_dim=input_channels, rnn_type=model_type)
    else:
        raise ValueError(f"Unknown model_type {model_type}, choose from cnn, lstm, rnn, gru")


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

    print(f"[INFO] Initializing '{model_args.model_type.upper()}' baseline architecture...")
    model = get_model(
        model_args.model_type,
        model_args.input_channels,
        seq_len=train_ds.seq.shape[1],
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

    print("Trainer device:", trainer.args.device)

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
