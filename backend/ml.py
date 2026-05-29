"""Train classifiers (CNN/LSTM/GLM/XGBoost) and run inference."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from backend.config import DEFAULT_WINDOW_SAMPLES
from backend.features import (
    DEFAULT_INPUT_FEATURES,
    feature_set_key,
    n_channels_for_features,
    normalize_feature_list,
)
from backend.metrics import classification_metrics
from backend.training_windows import TrainingWindow

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent / "data_pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from models import CNN, LSTM  # noqa: E402

ModelType = Literal["cnn", "lstm", "glm", "xgboost"]
ModelMode = Literal["none", "s3_ptl", "train"]
TORCH_MODEL_TYPES = frozenset({"cnn", "lstm"})
TABULAR_MODEL_TYPES = frozenset({"glm", "xgboost"})


@dataclass
class ModelRunConfig:
    mode: ModelMode = "none"
    model_type: ModelType = "cnn"
    # S3 engine: explicit key or auto-match per night
    engine_s3_key: Optional[str] = None
    auto_match_engine: bool = True
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    test_frac: float = 0.2
    early_stopping_patience: int = 5
    device: str = "cpu"
    input_features: Tuple[str, ...] = DEFAULT_INPUT_FEATURES
    window_samples: int = DEFAULT_WINDOW_SAMPLES
    # GLM (sklearn LogisticRegression)
    glm_C: float = 1.0
    glm_max_iter: int = 1000
    glm_penalty: str = "l2"
    glm_solver: str = "lbfgs"
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 1.0

    @property
    def n_channels(self) -> int:
        return n_channels_for_features(self.input_features)

    @property
    def feature_set_key(self) -> str:
        return feature_set_key(self.input_features)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], training: Optional[Dict[str, Any]] = None) -> "ModelRunConfig":
        train = training or {}
        raw_features = d.get("input_features", train.get("input_features"))
        return cls(
            mode=d.get("mode", "none"),
            model_type=d.get("architecture", d.get("model_type", "cnn")),
            engine_s3_key=d.get("engine_s3_key"),
            auto_match_engine=bool(d.get("auto_match_engine", True)),
            epochs=int(d.get("epochs", 5)),
            batch_size=int(d.get("batch_size", 32)),
            learning_rate=float(d.get("learning_rate", 0.001)),
            test_frac=float(d.get("test_frac", 0.2)),
            early_stopping_patience=int(d.get("early_stopping_patience", 5)),
            device=d.get("device", "cpu"),
            input_features=normalize_feature_list(raw_features),
            window_samples=int(d.get("window_samples", DEFAULT_WINDOW_SAMPLES)),
            glm_C=float(d.get("glm_C", 1.0)),
            glm_max_iter=int(d.get("glm_max_iter", 1000)),
            glm_penalty=str(d.get("glm_penalty", "l2")),
            glm_solver=str(d.get("glm_solver", "lbfgs")),
            xgb_n_estimators=int(d.get("xgb_n_estimators", 100)),
            xgb_max_depth=int(d.get("xgb_max_depth", 6)),
            xgb_learning_rate=float(d.get("xgb_learning_rate", 0.1)),
            xgb_subsample=float(d.get("xgb_subsample", 0.8)),
            xgb_colsample_bytree=float(d.get("xgb_colsample_bytree", 0.8)),
            xgb_min_child_weight=float(d.get("xgb_min_child_weight", 1.0)),
        )


def is_tabular_model(model_type: str) -> bool:
    return model_type in TABULAR_MODEL_TYPES


def is_torch_model(model_type: str) -> bool:
    return model_type in TORCH_MODEL_TYPES


def prepare_sequence(
    seq: Any,
    length: int = DEFAULT_WINDOW_SAMPLES,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    arr = np.asarray(seq, dtype=dtype)
    if arr.ndim == 2:
        # (n_channels, time) — pad/trim time axis per channel
        n_ch, t = arr.shape
        if t < length:
            pad = np.zeros((n_ch, length - t), dtype=dtype)
            arr = np.concatenate([arr, pad], axis=1)
        elif t > length:
            arr = arr[:, :length]
        if dtype == np.float32:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr
    flat = arr.flatten()
    if len(flat) < length:
        flat = np.pad(flat, (0, length - len(flat)))
    elif len(flat) > length:
        flat = flat[:length]
    if dtype == np.float32:
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return flat


def _window_array(seq: Any, length: int, dtype: np.dtype) -> np.ndarray:
    arr = prepare_sequence(seq, length, dtype=dtype)
    if arr.ndim == 1:
        return arr.reshape(1, length)
    return arr


def windows_to_samples(
    windows: List[TrainingWindow],
    length: int = DEFAULT_WINDOW_SAMPLES,
    *,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    if not windows:
        return np.zeros((0, 1, length), dtype=dtype), np.zeros(0, dtype=np.int64)
    stacked = [_window_array(w.sequence, length, dtype) for w in windows]
    n_ch = stacked[0].shape[0]
    x = np.stack(stacked, axis=0)  # (N, C, T)
    if n_ch == 1:
        x = x[:, 0, :]  # (N, T) legacy single-channel
    y = np.array([w.label for w in windows], dtype=np.int64)
    return x, y


def windows_to_feature_matrix(x: np.ndarray) -> np.ndarray:
    """Flatten window arrays to (N, features) for tabular classifiers."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        n, c, t = arr.shape
        return arr.reshape(n, c * t)
    raise ValueError(f"Expected 1–3D window array, got shape {arr.shape}")


def predict_tabular_proba(model: Any, x: np.ndarray) -> np.ndarray:
    """Return (N, 3) class probabilities for GLM/XGBoost models."""
    X = windows_to_feature_matrix(x)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    proba = model.predict_proba(X)
    if proba.shape[1] == 3:
        return proba
    # Align to 3 classes if a label was missing in training
    aligned = np.zeros((proba.shape[0], 3), dtype=np.float64)
    for i, cls in enumerate(model.classes_):
        if int(cls) < 3:
            aligned[:, int(cls)] = proba[:, i]
    return aligned


def _jit_batch_tensor(
    x: np.ndarray,
    device: str = "cpu",
    n_channels: int = 1,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> torch.Tensor:
    """
    Build input for mobile .ptl CNN engines.

    Production export traces CNN with float32 example shape (1, 1, 120).
    XNNPACK (used by optimize_for_mobile) rejects float64 inputs.
    """
    if n_channels != 1:
        raise ValueError(
            f"S3 .ptl engines support 1 input channel only; got {n_channels}. "
            "Use default features (audio) or train a new model."
        )
    x = np.ascontiguousarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    batch = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    if batch.dim() == 2:
        batch = batch.view(-1, 1, window_samples)
    elif batch.dim() == 3 and batch.shape[1] != 1:
        batch = batch.view(-1, 1, window_samples)
    return batch.contiguous()


class _WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, model_type: ModelType, n_channels: int = 1):
        self.x = torch.tensor(x, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.long)
        self.model_type = model_type
        self.n_channels = n_channels

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        seq = self.x[idx]
        if self.model_type == "lstm":
            if seq.dim() == 1:
                seq = seq.unsqueeze(-1)  # (T, 1)
            else:
                seq = seq.transpose(0, 1)  # (C, T) -> (T, C)
        elif self.model_type == "cnn" and seq.dim() == 1:
            pass  # (T,) single channel — forward reshapes
        return seq, self.y[idx]


def _make_model(model_type: ModelType, device: str, n_channels: int = 1) -> nn.Module:
    if model_type == "cnn":
        model = CNN(input_size=n_channels, output_size=3).double()
    else:
        model = LSTM(input_size=n_channels, output_size=3).double()
    return model.to(device)


def _forward(
    model: nn.Module,
    batch_x: torch.Tensor,
    model_type: ModelType,
    is_jit: bool = False,
    n_channels: int = 1,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> torch.Tensor:
    if is_jit:
        # Mobile .ptl: float32 (N, 1, T) on CPU — see _jit_batch_tensor
        if batch_x.dtype != torch.float32:
            batch_x = batch_x.float()
        if batch_x.dim() == 2:
            batch_x = batch_x.view(-1, 1, window_samples)
        return model(batch_x.contiguous())
    if model_type == "cnn":
        if batch_x.dim() == 2:
            batch_x = batch_x.unsqueeze(1)  # (N, T) -> (N, 1, T)
        elif batch_x.dim() == 3 and batch_x.shape[1] != n_channels:
            if batch_x.shape[-1] == n_channels:
                batch_x = batch_x.transpose(1, 2)
        return model(batch_x)
    if batch_x.dim() == 2:
        batch_x = batch_x.unsqueeze(-1)
    elif batch_x.dim() == 3:
        batch_x = batch_x.transpose(1, 2)  # (N, C, T) -> (N, T, C)
    return model(batch_x)


def _validate_training_windows(
    train_windows: List[TrainingWindow],
    val_windows: List[TrainingWindow],
    config: ModelRunConfig,
) -> None:
    if len(val_windows) < 1:
        raise ValueError(f"Need at least 1 val window; got {len(val_windows)}.")
    if is_tabular_model(config.model_type):
        if len(train_windows) < 3:
            raise ValueError(
                f"Need at least 3 train windows for {config.model_type}; got {len(train_windows)}."
            )
    elif len(train_windows) < config.batch_size:
        raise ValueError(
            f"Need at least {config.batch_size} train windows; got {len(train_windows)}."
        )


def _finalize_train_metrics(
    model: Any,
    val_windows: List[TrainingWindow],
    config: ModelRunConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = evaluate_windows(
        model,
        val_windows,
        config.model_type,
        device=config.device,
        is_jit=False,
        n_channels=config.n_channels,
        window_samples=config.window_samples,
        batch_size=config.batch_size,
    )
    metrics["model_type"] = config.model_type
    metrics["mode"] = "train"
    metrics["input_features"] = list(config.input_features)
    metrics["feature_set_key"] = config.feature_set_key
    if extra:
        metrics.update(extra)
    return metrics


def _train_torch_model(
    train_windows: List[TrainingWindow],
    val_windows: List[TrainingWindow],
    config: ModelRunConfig,
) -> Tuple[Any, Dict[str, Any]]:
    device = config.device
    n_ch = config.n_channels
    model = _make_model(config.model_type, device, n_channels=n_ch)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=config.learning_rate)

    win_len = config.window_samples
    x_tr, y_tr = windows_to_samples(train_windows, length=win_len)
    x_va, y_va = windows_to_samples(val_windows, length=win_len)
    train_loader = DataLoader(
        _WindowDataset(x_tr, y_tr, config.model_type, n_channels=n_ch),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        _WindowDataset(x_va, y_va, config.model_type, n_channels=n_ch),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_val = float("inf")
    patience = 0
    history: List[Dict[str, float]] = []

    model.train()
    for epoch in range(config.epochs):
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optim.zero_grad()
            pred = _forward(
                model,
                batch_x.to(device),
                config.model_type,
                n_channels=n_ch,
                window_samples=win_len,
            )
            loss = criterion(pred, batch_y.to(device))
            loss.backward()
            optim.step()
            batch_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = _forward(
                    model,
                    batch_x.to(device),
                    config.model_type,
                    n_channels=n_ch,
                    window_samples=win_len,
                )
                val_losses.append(criterion(pred, batch_y.to(device)).item())
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": avg_val})

        if avg_val < best_val - 1e-3:
            best_val = avg_val
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
        model.train()

    model.eval()
    metrics = _finalize_train_metrics(model, val_windows, config, {"training_history": history})
    return model, metrics


def _train_glm_model(
    train_windows: List[TrainingWindow],
    val_windows: List[TrainingWindow],
    config: ModelRunConfig,
) -> Tuple[Any, Dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    win_len = config.window_samples
    x_tr, y_tr = windows_to_samples(train_windows, length=win_len)
    X = windows_to_feature_matrix(x_tr)
    model = LogisticRegression(
        C=config.glm_C,
        max_iter=config.glm_max_iter,
        penalty=config.glm_penalty,
        solver=config.glm_solver,
        random_state=0,
    )
    model.fit(X, y_tr)
    metrics = _finalize_train_metrics(
        model,
        val_windows,
        config,
        {
            "glm_C": config.glm_C,
            "glm_max_iter": config.glm_max_iter,
            "glm_penalty": config.glm_penalty,
            "glm_solver": config.glm_solver,
        },
    )
    return model, metrics


def _train_xgboost_model(
    train_windows: List[TrainingWindow],
    val_windows: List[TrainingWindow],
    config: ModelRunConfig,
) -> Tuple[Any, Dict[str, Any]]:
    from xgboost import XGBClassifier

    win_len = config.window_samples
    x_tr, y_tr = windows_to_samples(train_windows, length=win_len)
    X = windows_to_feature_matrix(x_tr)
    model = XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        min_child_weight=config.xgb_min_child_weight,
        objective="multi:softprob",
        num_class=3,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X, y_tr)
    metrics = _finalize_train_metrics(
        model,
        val_windows,
        config,
        {
            "xgb_n_estimators": config.xgb_n_estimators,
            "xgb_max_depth": config.xgb_max_depth,
            "xgb_learning_rate": config.xgb_learning_rate,
            "xgb_subsample": config.xgb_subsample,
            "xgb_colsample_bytree": config.xgb_colsample_bytree,
            "xgb_min_child_weight": config.xgb_min_child_weight,
        },
    )
    return model, metrics


def train_model(
    train_windows: List[TrainingWindow],
    val_windows: List[TrainingWindow],
    config: ModelRunConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a classifier on window samples."""
    _validate_training_windows(train_windows, val_windows, config)
    if config.model_type == "glm":
        return _train_glm_model(train_windows, val_windows, config)
    if config.model_type == "xgboost":
        return _train_xgboost_model(train_windows, val_windows, config)
    return _train_torch_model(train_windows, val_windows, config)


def predict_windows(
    model: Any,
    windows: List[TrainingWindow],
    model_type: ModelType,
    device: str = "cpu",
    is_jit: bool = True,
    batch_size: int = 32,
    n_channels: int = 1,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> Tuple[List[int], List[List[float]]]:
    """Return class predictions and probability vectors per window."""
    if not windows:
        return [], []

    sample_dtype = np.float32 if is_jit else np.float64
    x, _ = windows_to_samples(windows, length=window_samples, dtype=sample_dtype)
    preds: List[int] = []
    probs: List[List[float]] = []

    if is_tabular_model(model_type):
        for start in range(0, len(x), batch_size):
            chunk = x[start : start + batch_size]
            p = predict_tabular_proba(model, chunk)
            pred_cls = np.argmax(p, axis=1)
            preds.extend(pred_cls.tolist())
            probs.extend(p.tolist())
        return preds, probs

    run_device = "cpu" if is_jit else device

    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = x[start : start + batch_size]
            if is_jit:
                batch = _jit_batch_tensor(
                    chunk,
                    device=run_device,
                    n_channels=n_channels,
                    window_samples=window_samples,
                )
            else:
                batch = torch.tensor(chunk, dtype=torch.float64, device=run_device)
            out = _forward(
                model,
                batch,
                model_type,
                is_jit=is_jit,
                n_channels=n_channels,
                window_samples=window_samples,
            )
            if out.dim() == 1:
                out = out.unsqueeze(0)
            p = out.detach().cpu().numpy()
            pred_cls = np.argmax(p, axis=1)
            preds.extend(pred_cls.tolist())
            probs.extend(p.tolist())
    return preds, probs


def evaluate_windows(
    model: Any,
    windows: List[TrainingWindow],
    model_type: ModelType,
    device: str = "cpu",
    is_jit: bool = True,
    n_channels: int = 1,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Classification metrics on held-out windows."""
    if not windows:
        return {"n_samples": 0}
    run_device = "cpu" if is_jit else device
    preds, _ = predict_windows(
        model,
        windows,
        model_type,
        device=run_device,
        is_jit=is_jit,
        n_channels=n_channels,
        window_samples=window_samples,
        batch_size=batch_size,
    )
    labels = [w.label for w in windows]
    metrics = classification_metrics(
        preds,
        labels,
        class_names=["non-apnea", "onset-apnea", "flatline"],
    )
    metrics["mode"] = "s3_ptl" if is_jit else "train"
    return metrics
