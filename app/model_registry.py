"""
model_registry.py — Model loading and caching.

ModelRegistry loads both LSTM models exactly once at startup, caches them,
and exposes a clean interface so route handlers never touch PyTorch directly.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

log = logging.getLogger("prism.model_registry")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths — overridable via env vars so Render/Docker can relocate them
_BASE = Path(os.getenv("MODEL_DIR", "."))
US_MODEL_PATH    = Path(os.getenv("US_MODEL_PATH",    str(_BASE / "us_stock_lstm.pth")))
INDIA_MODEL_PATH = Path(os.getenv("INDIA_MODEL_PATH", str(_BASE / "multi_stock_lstm_v2.pth")))

# Hyper-params must match training
US_SEQ_LEN     = 30
US_N_FEATURES  = 26
US_HIDDEN_SIZE = 128
US_N_LAYERS    = 2
US_EMBED_DIM   = 12
US_DROPOUT     = 0.35

IN_SEQ_LEN     = 30
IN_N_FEATURES  = 33
IN_HIDDEN_SIZE = 128
IN_N_LAYERS    = 2
IN_EMBED_DIM   = 12
IN_DROPOUT     = 0.35

VERSION = os.getenv("APP_VERSION", "1.0.0")
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")


# ── Model architectures (must match training exactly) ─────────────────────────

class MultiStockLSTM(nn.Module):
    """US multi-stock LSTM with per-stock learned embeddings."""

    def __init__(
        self,
        n_stocks: int,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed   = nn.Embedding(n_stocks, embed_dim)
        self.lstm    = nn.LSTM(
            input_size  = n_features + embed_dim,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed(stock_ids)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x   = torch.cat([x, emb], dim=-1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class IndiaMultiStockLSTM(nn.Module):
    """India v2 multi-stock LSTM with projection layer and deeper head."""

    def __init__(
        self,
        n_stocks: int  = 29,
        n_feat: int    = 33,
        embed_dim: int = 12,
        hidden: int    = 128,
        n_layers: int  = 2,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.emb  = nn.Embedding(n_stocks, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(n_feat + embed_dim, hidden),
            nn.LayerNorm(hidden),
        )
        self.lstm = nn.LSTM(
            input_size  = hidden,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.ln   = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        emb = self.emb(stock_ids)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x   = torch.cat([x, emb], dim=-1)
        x   = self.proj(x)
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        return self.head(out).squeeze(-1)


# ── Registry ──────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Singleton that loads both models once and exposes them for inference.

    Usage:
        registry = ModelRegistry.get()
        model    = registry.get_model("US")
        meta     = registry.get_meta("US")
    """

    _instance: "ModelRegistry | None" = None

    def __init__(self, n_us_stocks: int, n_in_stocks: int) -> None:
        self._us_model_ok = False
        self._in_model_ok = False

        self.us_model, self._us_model_ok = self._load_us_model(n_us_stocks)
        self.us_loaded_at = datetime.now(timezone.utc).isoformat()

        self.in_model, self.in_scalers, self._in_model_ok = self._load_india_model(n_in_stocks)
        self.in_loaded_at = datetime.now(timezone.utc).isoformat()

        self.meta_us: dict = {
            "model_name":    "multi-stock-lstm-us",
            "model_version": "v1",
            "framework":     "pytorch",
            "architecture": {
                "hidden":   US_HIDDEN_SIZE,
                "n_layers": US_N_LAYERS,
                "embed_dim": US_EMBED_DIM,
                "dropout":  US_DROPOUT,
                "n_feat":   US_N_FEATURES,
            },
            "seq_len":    US_SEQ_LEN,
            "n_stocks":   n_us_stocks,
            "model_file": US_MODEL_PATH.name,
            "device":     str(DEVICE),
            "market":     "US",
            "loaded_at":  self.us_loaded_at,
            "git_commit": GIT_COMMIT,
        }

        self.meta_in: dict = {
            "model_name":    "multi-stock-lstm-india",
            "model_version": "v1",
            "framework":     "pytorch",
            "architecture": {
                "hidden":   IN_HIDDEN_SIZE,
                "n_layers": IN_N_LAYERS,
                "embed_dim": IN_EMBED_DIM,
                "dropout":  IN_DROPOUT,
                "n_feat":   IN_N_FEATURES,
            },
            "seq_len":    IN_SEQ_LEN,
            "n_stocks":   n_in_stocks,
            "model_file": INDIA_MODEL_PATH.name,
            "device":     str(DEVICE),
            "market":     "IN",
            "loaded_at":  self.in_loaded_at,
            "git_commit": GIT_COMMIT,
        }

        # Enrich with model_metadata.json if available
        meta_file = Path("model_metadata.json")
        if meta_file.exists():
            try:
                import json
                with meta_file.open("r", encoding="utf-8") as f:
                    all_meta = json.load(f)
                    if "us_model" in all_meta:
                        self.meta_us["training_metrics"] = all_meta["us_model"].get("training_metrics", {})
                    if "india_model" in all_meta:
                        self.meta_in["training_metrics"] = all_meta["india_model"].get("training_metrics", {})
            except Exception as exc:
                log.warning("Could not load model_metadata.json: %s", exc)

    @classmethod
    def get(cls, n_us_stocks: int = 30, n_in_stocks: int = 29) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls(n_us_stocks, n_in_stocks)
        return cls._instance

    @property
    def us_ok(self) -> bool:
        return self._us_model_ok

    @property
    def in_ok(self) -> bool:
        return self._in_model_ok

    @property
    def all_ok(self) -> bool:
        return self._us_model_ok and self._in_model_ok

    # ── Loaders ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_truthy_env(var_name: str, default: str = "true") -> bool:
        return os.getenv(var_name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _load_us_model(self, n_stocks: int) -> tuple[MultiStockLSTM, bool]:
        net = MultiStockLSTM(
            n_stocks    = n_stocks,
            n_features  = US_N_FEATURES,
            hidden_size = US_HIDDEN_SIZE,
            n_layers    = US_N_LAYERS,
            embed_dim   = US_EMBED_DIM,
            dropout     = US_DROPOUT,
        ).to(DEVICE)

        if not US_MODEL_PATH.exists():
            log.warning("US model %s not found. Using random weights.", US_MODEL_PATH)
            net.eval()
            return net, False

        loaded = False
        try:
            state = torch.load(US_MODEL_PATH, map_location=DEVICE, weights_only=True)
            if isinstance(state, dict):
                state = state.get("model_state_dict", state.get("state_dict", state))
            net.load_state_dict(state, strict=False)
            loaded = True
            log.info("✓ US model loaded from %s", US_MODEL_PATH)
        except Exception:
            if self._is_truthy_env("TRUSTED_MODEL_CHECKPOINT", "true"):
                try:
                    state = torch.load(US_MODEL_PATH, map_location=DEVICE, weights_only=False)
                    if isinstance(state, dict):
                        state = state.get("model_state_dict", state.get("state_dict", state))
                    net.load_state_dict(state, strict=False)
                    loaded = True
                    log.info("✓ US model loaded (trusted) from %s", US_MODEL_PATH)
                except Exception as exc:
                    log.warning("US model load failed: %s. Using random weights.", exc)

        net.eval()
        return net, loaded

    def _load_india_model(self, n_stocks: int) -> tuple[IndiaMultiStockLSTM, dict, bool]:
        net = IndiaMultiStockLSTM(
            n_stocks  = n_stocks,
            n_feat    = IN_N_FEATURES,
            embed_dim = IN_EMBED_DIM,
            hidden    = IN_HIDDEN_SIZE,
            n_layers  = IN_N_LAYERS,
            dropout   = IN_DROPOUT,
        ).to(DEVICE)
        scalers: dict = {}

        if not INDIA_MODEL_PATH.exists():
            log.warning("India model %s not found. Using random weights.", INDIA_MODEL_PATH)
            net.eval()
            return net, scalers, False

        loaded = False
        try:
            ckpt = torch.load(INDIA_MODEL_PATH, map_location=DEVICE, weights_only=False)
            model_state = ckpt.get("model_state", ckpt)
            if isinstance(model_state, dict) and "emb.weight" in model_state:
                net.load_state_dict(model_state, strict=False)
                loaded = True
                log.info("✓ India model loaded from %s", INDIA_MODEL_PATH)
            else:
                log.warning("India checkpoint has unexpected structure.")

            saved_scalers = ckpt.get("scalers", {})
            if isinstance(saved_scalers, dict):
                scalers = saved_scalers
                log.info("  → %d per-ticker scalers loaded", len(scalers))
        except Exception as exc:
            log.warning("India model load failed: %s. Using random weights.", exc)

        net.eval()
        return net, scalers, loaded

    # ── Public interface ──────────────────────────────────────────────────────

    def get_model(self, market: str):
        return self.in_model if market == "IN" else self.us_model

    def get_meta(self, market: str) -> dict:
        return self.meta_in if market == "IN" else self.meta_us

    def get_scaler(self, symbol: str) -> RobustScaler | None:
        """Return saved scaler for an India symbol, or None."""
        return self.in_scalers.get(symbol)
