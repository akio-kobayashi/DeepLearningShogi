import inspect
import re
from importlib import import_module
from typing import Match, Type

import torch
import torch.nn as nn


_NETWORK_PATTERN = re.compile(
    r"^(.*?)(\d+)(x\d+){0,1}(_fcl\d+){0,1}(_reduction\d+){0,1}(_.+){0,1}$"
)
_DEFAULT_CHANNELS_BY_BLOCKS = {10: 192, 15: 224, 20: 256, 30: 384}
_ACTIVATIONS = {"_relu": nn.ReLU, "_swish": nn.SiLU}
_SPECIAL_PUBLISHED_NETWORKS = {"wideresnet10", "resnet10_swish"}


def _resolve_user_defined_network_class(network: str):
    names = network.split(".")
    if len(names) == 1:
        cls = globals().get(names[0])
    else:
        module = import_module(".".join(names[:-1]))
        cls = getattr(module, names[-1], None)
    if cls is None:
        raise ValueError(f"Unknown network class: {network}")
    if not inspect.isclass(cls):
        raise TypeError(f"Network target is not a class: {network}")
    return cls


def _resolve_policy_value_network_class(
    network: str,
    parsed: Match[str] | None,
) -> Type[nn.Module]:
    # wideresnet10 and resnet10_swish are treated specially because there are published models
    if network == "wideresnet10":
        from dlshogi.network.policy_value_network_wideresnet10 import PolicyValueNetwork
    elif network == "resnet10_swish":
        from dlshogi.network.policy_value_network_resnet10_swish import PolicyValueNetwork
    elif parsed:
        module_name = f"dlshogi.network.policy_value_network_{parsed[1]}"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as e:
            raise ValueError(f"Unknown network family '{parsed[1]}' in '{network}'") from e
        PolicyValueNetwork = getattr(module, "PolicyValueNetwork", None)
        if PolicyValueNetwork is None:
            raise ValueError(f"{module_name} does not define PolicyValueNetwork")
    else:
        PolicyValueNetwork = _resolve_user_defined_network_class(network)
    return PolicyValueNetwork


def _wrap_with_sigmoid(base_cls: Type[nn.Module]) -> Type[nn.Module]:
    class PolicyValueNetworkAddSigmoid(base_cls):
        def __init__(self, *args, **kwargs):
            super(PolicyValueNetworkAddSigmoid, self).__init__(*args, **kwargs)

        def forward(self, x1, x2):
            y1, y2 = super(PolicyValueNetworkAddSigmoid, self).forward(x1, x2)
            return y1, torch.sigmoid(y2)

    return PolicyValueNetworkAddSigmoid


def _build_pattern_network_kwargs(parsed: Match[str]):
    blocks = int(parsed[2])
    if parsed[3] is None and blocks not in _DEFAULT_CHANNELS_BY_BLOCKS:
        raise ValueError(
            f"Unsupported block count '{blocks}' without explicit channels. "
            f"Use e.g. '{parsed[1]}{blocks}x256'."
        )
    channels = int(parsed[3][1:]) if parsed[3] is not None else _DEFAULT_CHANNELS_BY_BLOCKS[blocks]
    fcl = int(parsed[4][4:]) if parsed[4] is not None else 256
    if parsed[6] is not None and parsed[6] not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation suffix '{parsed[6]}'. "
            f"Expected one of: {', '.join(sorted(_ACTIVATIONS))}"
        )
    activation = _ACTIVATIONS[parsed[6]]() if parsed[6] is not None else nn.ReLU()
    kwargs = {"blocks": blocks, "channels": channels, "activation": activation, "fcl": fcl}
    if parsed[1] == "senet":
        kwargs["reduction"] = int(parsed[5][10:]) if parsed[5] is not None else 8
    return kwargs


def policy_value_network(network: str, add_sigmoid: bool = False) -> nn.Module:
    # Parallel-introduction note:
    # - Legacy families are kept as-is.
    # - New families (e.g. "resnetx10") can be added side-by-side.
    parsed = _NETWORK_PATTERN.match(network)
    policy_value_network_cls = _resolve_policy_value_network_class(network, parsed)

    if add_sigmoid:
        policy_value_network_cls = _wrap_with_sigmoid(policy_value_network_cls)

    if network in _SPECIAL_PUBLISHED_NETWORKS:
        return policy_value_network_cls()
    if parsed:
        return policy_value_network_cls(**_build_pattern_network_kwargs(parsed))
    return policy_value_network_cls()
