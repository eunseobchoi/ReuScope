"""Compatibility alias for the official Kvasir-Capsule split loader."""

import sys

from utils import dataset_kvasir_official as _impl

sys.modules[__name__] = _impl
