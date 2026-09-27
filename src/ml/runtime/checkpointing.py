from collections.abc import Callable

from torch.utils.checkpoint import checkpoint


def activation_checkpoint(module: Callable, *args, enabled: bool = True, **kwargs):
    if not enabled:
        return module(*args, **kwargs)
    return checkpoint(module, *args, use_reentrant=False, **kwargs)
