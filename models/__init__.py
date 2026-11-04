"""Model registry used by evaluate_detailed.py and any other consumer."""
from .custom_cnn import ECGCNN
from .mobilenet import MobileNetDerivation, FusionModel

ARCH_REGISTRY = {
    'ECGCNN': ECGCNN,
    'MobileNetDerivation': MobileNetDerivation,
    'FusionModel': FusionModel,
}

__all__ = ['ECGCNN', 'MobileNetDerivation', 'FusionModel', 'ARCH_REGISTRY']
