from typing import List, Dict, Any

import torch

from torch_geometric.nn import Sequential
from torch_geometric.typing import NodeType, EdgeType

from torch_frame import stype
from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import (
    BlueprintModel,
    MeanSumConv,
    ResidualNorm,
    SelfAttention,
)

from .utils import get_decoder, get_encoder


def create_tabtransformer_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    gnn_layers = config.get("gnn_layers", 1)
    mlp_dims = config.get("mlp_dims", [])
    num_heads = config.get("num_heads", 1)
    batch_norm = config.get("batch_norm", False)
    dropout = config.get("dropout", 0)

    is_classification = defaults.task == TaskType.CLASSIFICATION

    output_dim = (
        len(col_stats_dict[target[0]][target[1]][StatType.COUNT][0])
        if is_classification
        else 1
    )

    def get_cat_num_split(node):
        num_num_cols = len(col_names_dict[node].get(stype.numerical, []))
        return lambda x: (x[:, :num_num_cols], x[:, num_num_cols:])

    return BlueprintModel(
        target=target,
        embed_dim=embed_dim,
        col_stats_per_table=col_stats_dict,
        col_names_dict_per_table=col_names_dict,
        edge_types=edge_types,
        stype_encoder_dict=get_encoder("tabtransformer"),
        positional_encoding=False,
        num_gnn_layers=gnn_layers,
        pre_combination=lambda i, node, cols: Sequential(
            "x_in_raw",
            [
                (get_cat_num_split(node), "x_in_raw -> x_num, x_in"),
                (
                    SelfAttention(embed_dim, num_heads, dropout=dropout),
                    "x_in -> x_next",
                ),
                (ResidualNorm(embed_dim), "x_in, x_next -> x_in"),
                (
                    torch.nn.Sequential(
                        torch.nn.Linear(embed_dim, embed_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(embed_dim, embed_dim),
                    ),
                    "x_in -> x_next",
                ),
                (ResidualNorm(embed_dim), "x_in, x_next -> x_out"),
                (
                    lambda x_num, x_cat: torch.concat([x_num, x_cat], dim=1),
                    "x_num, x_out -> x_out",
                ),
            ],
        ),
        table_combination=lambda i, edge, cols: MeanSumConv(),
        decoder_aggregation=lambda x: x.view(*x.shape[:-2], -1),
        decoder=lambda cols: get_decoder(
            len(cols) * embed_dim,
            output_dim,
            mlp_dims,
            batch_norm,
        ),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        positional_encoding_dropout=0.0,
    )
