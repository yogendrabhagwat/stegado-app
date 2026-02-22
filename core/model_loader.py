"""
Centralized Model Loader.
Tries to load pretrained PyTorch models; gracefully falls back to
the robust signal-processing engine when models are unavailable.
"""

import os
import logging

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_encoder_cache = {}
_decoder_cache = {}


def _model_path(name: str) -> str:
    return os.path.join(_MODELS_DIR, f'{name}.pth')


def load_universal_encoder():
    if 'universal_encoder' in _encoder_cache:
        return _encoder_cache['universal_encoder']
    if not TORCH_AVAILABLE:
        return None
    try:
        from core.universal_model import UniversalEncoder
        model = UniversalEncoder(secret_dim=256)
        path = _model_path('universal_encoder')
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
            logger.info('Loaded pretrained universal encoder.')
        else:
            model.eval()
            logger.info('Universal encoder initialized (untrained – using fallback).')
        _encoder_cache['universal_encoder'] = model
        return model
    except Exception as e:
        logger.warning(f'Could not load universal encoder: {e}')
        return None


def load_universal_decoder():
    if 'universal_decoder' in _decoder_cache:
        return _decoder_cache['universal_decoder']
    if not TORCH_AVAILABLE:
        return None
    try:
        from core.universal_model import SecretDecoder
        model = SecretDecoder(secret_dim=256)
        path = _model_path('universal_decoder')
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
            logger.info('Loaded pretrained universal decoder.')
        else:
            model.eval()
        _decoder_cache['universal_decoder'] = model
        return model
    except Exception as e:
        logger.warning(f'Could not load universal decoder: {e}')
        return None


def load_discriminator():
    if not TORCH_AVAILABLE:
        return None
    try:
        from core.gan_discriminator import SteganalysisDiscriminator
        model = SteganalysisDiscriminator()
        path = _model_path('discriminator')
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
        return model
    except Exception as e:
        logger.warning(f'Could not load discriminator: {e}')
        return None


def get_active_model(mode: str = 'Universal') -> str:
    """Return human-readable model status string."""
    if not TORCH_AVAILABLE:
        return f'{mode} (Signal-Processing Engine)'
    path = _model_path('universal_encoder')
    if os.path.isfile(path):
        return f'{mode} (AI Neural Model)'
    return f'{mode} (Signal-Processing Engine)'
