"""Leakage-safe structured metadata encoding pipeline (timestamps, ordinal, one-hot, scaling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ticket_classification.domain.base_stage import LeakageSafePipelineStage


class MetadataFeaturePipeline(LeakageSafePipelineStage[pd.DataFrame, np.ndarray]):
    """Encodes structured ticket/claim metadata while strictly preventing train-test leakage."""

    def __init__(
        self,
        date_cols: list[str] | None = None,
        ordinal_cols: list[str] | None = None,
        nominal_cols: list[str] | None = None,
        numeric_cols: list[str] | None = None,
        max_onehot_categories: int = 20,
    ) -> None:
        super().__init__()
        self.date_cols = date_cols or []
        self.ordinal_cols = ordinal_cols or []
        self.nominal_cols = nominal_cols or []
        self.numeric_cols = numeric_cols or []
        self.max_onehot_categories = max_onehot_categories

        self._ordinal_encoder: OrdinalEncoder | None = None
        self._onehot_encoder: OneHotEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._numeric_medians: dict[str, float] = {}

    def _extract_numeric_and_dates(self, df: pd.DataFrame, is_fit: bool) -> np.ndarray:
        parts: list[np.ndarray] = []

        # Text length, token count, and missing text indicators
        text_series = df.get("raw_text", pd.Series([""] * len(df))).fillna("")
        char_len = text_series.str.len().to_numpy(dtype=float).reshape(-1, 1)
        word_count = (
            text_series.str.split().str.len().to_numpy(dtype=float).reshape(-1, 1)
        )
        is_text_empty = (
            (text_series.str.strip() == "").astype(float).to_numpy().reshape(-1, 1)
        )
        parts.extend([char_len, word_count, is_text_empty])

        # Date parsing and missing indicator generation
        for col in self.date_cols:
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce")
                is_missing = ts.isna().astype(float).to_numpy().reshape(-1, 1)
                ts_num = (ts.astype("int64") // 10**9).astype(float)
                ts_num = ts_num.where(~ts.isna(), 0.0).to_numpy().reshape(-1, 1)
                parts.extend([ts_num, is_missing])

        # Numerical imputation with missing indicators
        for col in self.numeric_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                is_missing = s.isna().astype(float).to_numpy().reshape(-1, 1)
                if is_fit:
                    med = float(s.median()) if not s.dropna().empty else 0.0
                    self._numeric_medians[col] = med
                med = self._numeric_medians.get(col, 0.0)
                filled = s.fillna(med).to_numpy(dtype=float).reshape(-1, 1)
                parts.extend([filled, is_missing])

        return np.hstack(parts) if parts else np.zeros((len(df), 1), dtype=float)

    def _fit_impl(self, train_data: pd.DataFrame) -> None:
        num_mat = self._extract_numeric_and_dates(train_data, is_fit=True)
        self._scaler = StandardScaler()
        self._scaler.fit(num_mat)

        valid_ord = [c for c in self.ordinal_cols if c in train_data.columns]
        if valid_ord:
            self._ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            ord_df = train_data[valid_ord].fillna("MISSING").astype(str)
            self._ordinal_encoder.fit(ord_df)

        valid_nom = [c for c in self.nominal_cols if c in train_data.columns]
        if valid_nom:
            self._onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                max_categories=self.max_onehot_categories,
                sparse_output=False,
            )
            nom_df = train_data[valid_nom].fillna("MISSING").astype(str)
            self._onehot_encoder.fit(nom_df)

    def _transform_impl(self, data: pd.DataFrame) -> np.ndarray:
        assert self._scaler is not None
        num_mat = self._extract_numeric_and_dates(data, is_fit=False)
        blocks = [self._scaler.transform(num_mat)]

        valid_ord = [c for c in self.ordinal_cols if c in data.columns]
        if self._ordinal_encoder is not None and valid_ord:
            ord_df = data[valid_ord].fillna("MISSING").astype(str)
            blocks.append(self._ordinal_encoder.transform(ord_df))

        valid_nom = [c for c in self.nominal_cols if c in data.columns]
        if self._onehot_encoder is not None and valid_nom:
            nom_df = data[valid_nom].fillna("MISSING").astype(str)
            blocks.append(self._onehot_encoder.transform(nom_df))

        return np.hstack(blocks)
