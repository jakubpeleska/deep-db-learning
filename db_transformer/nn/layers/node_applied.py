from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch_geometric.data.data import NodeType


class NodeApplied(torch.nn.Module):
    def __init__(
        self,
        factory: Callable[[NodeType], Union[torch.nn.Module, Callable[[Any], Any]]],
        node_types: List[NodeType],
        learnable: bool = True,
        dynamic_args: bool = False,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.dynamic_args = dynamic_args

        self.node_layer_dict = {}
        for k in node_types:
            self.node_layer_dict[k] = factory(k)
            if not isinstance(self.node_layer_dict[k], torch.nn.Module):
                learnable = False

        if learnable:
            self.node_layer_dict = torch.nn.ModuleDict(self.node_layer_dict)

    def forward(self, x_dict: Dict[NodeType, Any], *argv) -> Dict[NodeType, Any]:
        out_dict: Dict[NodeType, Any] = {}

        if self.dynamic_args:
            input_dict = defaultdict(list)
            for k in x_dict:
                input_dict[k].append(x_dict[k])
                for arg in argv:
                    if not isinstance(arg, dict):
                        input_dict[k].append(arg)
                    else:
                        input_dict[k].append(arg[k])

            for k, xs in input_dict.items():
                out_dict[k] = self.node_layer_dict[k](*xs)
        else:
            for k, x in x_dict.items():
                out_dict[k] = self.node_layer_dict[k](x)

        return out_dict
