import torch
import torch.nn as nn
import inspect
import functools


def store_config_args(func):
    attrs, varargs, varkw, defaults = inspect.getfullargspec(func)[:4]

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val
        return func(self, *args, **kwargs)
    return wrapper


class LoadableModel(nn.Module):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model
