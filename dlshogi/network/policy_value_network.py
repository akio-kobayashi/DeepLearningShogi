import re
from importlib import import_module
from typing import Match

import torch
import torch.nn as nn


_NETWORK_PATTERN = re.compile(
    r"^(.*?)(\d+)(x\d+){0,1}(_fcl\d+){0,1}(_reduction\d+){0,1}(_.+){0,1}$"
)
_DEFAULT_CHANNELS_BY_BLOCKS = {10: 192, 15: 224, 20: 256, 30: 384}
_ACTIVATIONS = {"_relu": nn.ReLU, "_swish": nn.SiLU}
_SPECIAL_PUBLISHED_NETWORKS = {"wideresnet10", "resnet10_swish"}


def _resolve_policy_value_network_class(network: str, parsed: Match[str] | None):
    # wideresnet10 and resnet10_swish are treated specially because there are published models
    if network == "wideresnet10":
        from dlshogi.network.policy_value_network_wideresnet10 import PolicyValueNetwork
    elif network == "resnet10_swish":
        from dlshogi.network.policy_value_network_resnet10_swish import PolicyValueNetwork
    elif parsed:
        module = import_module(f"dlshogi.network.policy_value_network_{parsed[1]}")
        PolicyValueNetwork = getattr(module, "PolicyValueNetwork")
    else:
        # user defined network
        names = network.split(".")
        if len(names) == 1:
            PolicyValueNetwork = globals()[names[0]]
        else:
            PolicyValueNetwork = getattr(import_module(".".join(names[:-1])), names[-1])
    return PolicyValueNetwork


def _wrap_with_sigmoid(base_cls):
    class PolicyValueNetworkAddSigmoid(base_cls):
        def __init__(self, *args, **kwargs):
            super(PolicyValueNetworkAddSigmoid, self).__init__(*args, **kwargs)

        def forward(self, x1, x2):
            y1, y2 = super(PolicyValueNetworkAddSigmoid, self).forward(x1, x2)
            return y1, torch.sigmoid(y2)

    return PolicyValueNetworkAddSigmoid


def _build_pattern_network_kwargs(parsed: Match[str]):
    blocks = int(parsed[2])
    channels = (
        int(parsed[3][1:])
        if parsed[3] is not None
        else _DEFAULT_CHANNELS_BY_BLOCKS[blocks]
    )
    fcl = int(parsed[4][4:]) if parsed[4] is not None else 256
    activation = (
        _ACTIVATIONS[parsed[6]]() if parsed[6] is not None else nn.ReLU()
    )
    kwargs = {"blocks": blocks, "channels": channels, "activation": activation, "fcl": fcl}
    if parsed[1] == "senet":
        kwargs["reduction"] = int(parsed[5][10:]) if parsed[5] is not None else 8
    return kwargs


def policy_value_network(network, add_sigmoid=False):
    parsed = _NETWORK_PATTERN.match(network)
    policy_value_network_cls = _resolve_policy_value_network_class(network, parsed)

    if add_sigmoid:
        policy_value_network_cls = _wrap_with_sigmoid(policy_value_network_cls)

    if network in _SPECIAL_PUBLISHED_NETWORKS:
        return policy_value_network_cls()
    if parsed:
        return policy_value_network_cls(**_build_pattern_network_kwargs(parsed))
    return policy_value_network_cls()
