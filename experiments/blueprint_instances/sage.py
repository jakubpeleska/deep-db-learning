from typing import List, Dict, Any

import torch

from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.nn import conv

from torch_frame import stype
from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import BlueprintModel

from .utils import get_decoder, get_encoder


def create_sage_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    gnn_layers = config.get("gnn_layers", 1)
    batch_norm = config.get("batch_norm", False)
    mlp_dims = config.get("mlp_dims", [])

    is_classification = defaults.task == TaskType.CLASSIFICATION

    output_dim = (
        len(col_stats_dict[target[0]][target[1]][StatType.COUNT][0])
        if is_classification
        else 1
    )

    return BlueprintModel(
        target=target,
        embed_dim=embed_dim,
        col_stats_per_table=col_stats_dict,
        col_names_dict_per_table=col_names_dict,
        edge_types=edge_types,
        stype_encoder_dict=get_encoder("basic"),
        post_embedder=lambda node, cols: lambda x: x.view(*x.shape[:-2], -1),
        positional_encoding=False,
        num_gnn_layers=gnn_layers,
        pre_combination=lambda i, node, cols: (
            torch.nn.Identity()
            if i == 0
            else torch.nn.Sequential(
                (
                    torch.nn.BatchNorm1d(
                        len(cols) * int(embed_dim / 2**i),
                    )
                    if batch_norm
                    else torch.nn.Identity()
                ),
                torch.nn.ReLU(),
            )
        ),
        table_combination=lambda i, edge, cols: conv.SAGEConv(
            (
                len(cols[0]) * (embed_dim // 2**i),
                len(cols[1]) * (embed_dim // 2**i),
            ),
            len(cols[1]) * (embed_dim // 2 ** (i + 1)),
            aggr="sum",
        ),
        decoder_aggregation=torch.nn.Identity(),
        decoder=lambda cols: get_decoder(
            len(cols) * embed_dim // 2**gnn_layers,
            output_dim,
            mlp_dims,
            batch_norm,
            out_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        ),
        positional_encoding_dropout=0.0,
    )
