"""Template Method base class enforcing strict Fit-on-Train-Only (zero data leakage)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class LeakageSafePipelineStage(ABC, Generic[TInput, TOutput]):
    """Enforces that stateful preprocessing/clustering stages fit strictly on training data."""

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, train_data: TInput) -> LeakageSafePipelineStage[TInput, TOutput]:
        """Fit internal parameters strictly on the training split."""
        self._fit_impl(train_data)
        self._is_fitted = True
        return self

    def transform(self, data: TInput) -> TOutput:
        """Transform validation, test, or live data without mutating fitted state."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}.transform() called before fit(). "
                "Always fit on the training split first to prevent data leakage."
            )
        return self._transform_impl(data)

    def fit_transform(self, train_data: TInput) -> TOutput:
        """Fit on training data and return transformed training data."""
        self.fit(train_data)
        return self.transform(train_data)

    @abstractmethod
    def _fit_impl(self, train_data: TInput) -> None:
        """Subclass implementation of fitting on training data."""

    @abstractmethod
    def _transform_impl(self, data: TInput) -> TOutput:
        """Subclass implementation of inductive transformation."""
