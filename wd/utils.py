import torch

from clearml import Task


CHANNEL_PRETRAIN = {'R': 0, 'G': 1, 'B': 2}


class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]


def fmt_scale(prefix, scale):
    """
    format scale name

    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace('.', '')
    return f'{prefix}_{scale_str}x'


def load_weight_from_clearml(task_name, model_name='ckpt_best'):
    t = Task.get_task(task_name=task_name)
    model = t.models['output'][model_name]
    path = model.get_weights()
    return torch.load(path)


def load_checkpoint_module_fix(state_dict):
    if 'net' in state_dict:
        state_dict = state_dict['net']
    def remove_starts_with_module(x):
        return remove_starts_with_module(x[7:]) if x.startswith('module.') else x
    return {remove_starts_with_module(k): v for k, v in state_dict.items()}


def calculate_channel_to_load(channel_to_load, in_channels):
    if channel_to_load is None:
        channel_to_load = slice(in_channels)
    elif isinstance(channel_to_load, str):
        if channel_to_load == 'complete':
            return channel_to_load
        extra_channels = in_channels - 3 # R, G, B
        channel_to_load = [0, 1, 2] + [CHANNEL_PRETRAIN[channel_to_load] for _ in range(extra_channels)
        ]
    else:
        channel_to_load = [CHANNEL_PRETRAIN[x] for x in channel_to_load]
    return channel_to_load


def adapt_weights_of_first_operation(weights, first_operation, channels_to_load):
    weights[first_operation] = \
        weights[first_operation][:, channels_to_load]
    return weights