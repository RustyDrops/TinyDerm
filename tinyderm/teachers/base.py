"""Teacher interface for Phase 2 specialist-teacher distillation."""

from abc import ABC, abstractmethod
from typing import Any
import torch


class Teacher(ABC):
    """Abstract base class for high-capacity dermatology specialist teachers."""

    @abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Returns softened probability distribution over diagnostic classes."""
        pass

    @abstractmethod
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts dense feature representation from backbone."""
        pass
