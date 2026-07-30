from __future__ import annotations

from abc import ABC, abstractmethod


class Metric(ABC):
    """Small TorchMetrics-compatible stateful metric interface."""

    @abstractmethod
    def update(self, preds, target):
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def __call__(self, preds, target):
        self.update(preds, target)
        return self.compute()

    def forward(self, preds, target):
        return self(preds, target)

    def clone(self):
        import copy

        return copy.deepcopy(self)

    def to(self, *args, **kwargs):
        return self
