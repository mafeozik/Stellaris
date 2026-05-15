"""Нейросетевая модель — многослойный персептрон (MLP).

Архитектура соответствует проектной части ВКР:
    Input -> Linear(256) -> ReLU -> Dropout(0.2)
          -> Linear(128) -> ReLU -> Dropout(0.2)
          -> Linear(64)  -> ReLU -> Dropout(0.2)
          -> Linear(K)   (K = число целевых переменных)

Обучение: Adam, MSELoss, early stopping по валидационной выборке.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN,
    MLP_LR,
    MLP_PATIENCE,
    MLP_WEIGHT_DECAY,
    SEED,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CampaignMLP(nn.Module):
    """MLP для прогнозирования метрик рекламных кампаний.

    Args:
        input_dim: Размерность входного вектора (после one-hot/scaling).
        output_dim: Число целевых переменных.
        hidden_dims: Размерности скрытых слоёв.
        dropout: Вероятность зануления в Dropout-слоях.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        hidden_dims: Sequence[int] = MLP_HIDDEN,
        dropout: float = MLP_DROPOUT,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = MLP_EPOCHS,
    lr: float = MLP_LR,
    weight_decay: float = MLP_WEIGHT_DECAY,
    batch_size: int = MLP_BATCH_SIZE,
    patience: int = MLP_PATIENCE,
    verbose: bool = True,
) -> tuple[CampaignMLP, TrainHistory]:
    """Обучает MLP с early stopping по валидационной выборке.

    Returns:
        Кортеж (обученная модель с лучшими весами, история обучения).
    """
    # Фиксируем seed для воспроизводимости
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = get_device()
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1

    model = CampaignMLP(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = TrainHistory()
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running / n_seen
        history.train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        history.val_losses.append(val_loss)

        improved = val_loss < history.best_val_loss - 1e-6
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(
                f"  Epoch {epoch:3d}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}"
                f"{'  <-- best' if improved else ''}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict(model: CampaignMLP, X: np.ndarray) -> np.ndarray:
    """Предсказание MLP по numpy-массиву."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        out = model(Xt).cpu().numpy()
    return out
