"""
This code is adapted from https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/shapley.py.
We enhanced the parallel computation of Shapley values for a faster performance
"""

from itertools import combinations
import numpy as np
import copy
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Dataset, DataLoader
from scipy.special import comb
import torch


def graph_build_split(X, edge_index, edge_attr, node_mask: torch.Tensor):
    """ subgraph building through spliting the selected nodes from the original graph """
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    ret_edge_attr = edge_attr[edge_mask, :]
    return ret_X, ret_edge_index, ret_edge_attr


def graph_build_zero_filling(X, edge_index, edge_attr, node_mask: torch.Tensor):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)

    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)

    ret_edge_attr = edge_attr * edge_mask.int().unsqueeze(1)

    return ret_X, edge_index, ret_edge_attr


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError


class MarginalSubgraphDataset(Dataset):
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):

      self.num_nodes = data.num_nodes
      self.X = data.x
      self.edge_index = data.edge_index
      self.edge_attr = data.edge_attr
      self.smiles = data.smiles

      self.device = self.X.device

      self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
      self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
      self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index, exclude_graph_edge_attr = self.subgraph_build_func(self.X, self.edge_index, self.edge_attr, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index, include_graph_edge_attr = self.subgraph_build_func(self.X, self.edge_index, self.edge_attr, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index, edge_attr=exclude_graph_edge_attr)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index, edge_attr=include_graph_edge_attr)
        return exclude_data, include_data

    def get(self): pass

    def len(self): pass


def marginal_contribution(model, data: Data, exclude_mask: np.array, include_mask: np.array,
                          value_func, subgraph_build_func):
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """
    marginal_subgraph_dataset = MarginalSubgraphDataset(data, exclude_mask, include_mask, subgraph_build_func)
    dataloader = DataLoader(marginal_subgraph_dataset, batch_size=256, shuffle=False, num_workers=0)

    marginal_contribution_list = []

    for exclude_data, include_data in dataloader:
        exclude_values = value_func(model, exclude_data)
        include_values = value_func(model, include_data)
        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions

def prepare_single_coalition(coalition: list, local_region, num_nodes):

    set_exclude_masks = []
    set_include_masks = []
    nodes_around = [node for node in local_region if node not in coalition]
    num_nodes_around = len(nodes_around)

    for subset_len in range(0, num_nodes_around + 1):
        node_exclude_subsets = combinations(nodes_around, subset_len)
        for node_exclude_subset in node_exclude_subsets:
            set_exclude_mask = np.ones(num_nodes)
            set_exclude_mask[local_region] = 0.0
            if node_exclude_subset:
                set_exclude_mask[list(node_exclude_subset)] = 1.0
            set_include_mask = set_exclude_mask.copy()
            set_include_mask[coalition] = 1.0

            set_exclude_masks.append(set_exclude_mask)
            set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    num_players = len(nodes_around) + 1
    num_player_in_set = num_players - 1 + len(coalition) - (1 - exclude_mask).sum(axis=1)
    p = num_players
    S = num_player_in_set
    coeffs = list(1.0 / comb(p, S) / (p - S + 1e-6))

    return set_exclude_masks, set_include_masks, coeffs


def l_shapley_paralell(model, coalitions, data: Data, value_func: str, subgraph_building_method='zero_filling'):

    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    # local_region = [i for i in range(num_nodes)]

    set_exclude_masks = []
    set_include_masks = []
    set_coeffs = []
    coalition_batch = []

    for coalition in coalitions:

        local_region = copy.copy(coalition)
        local_radius = 4
        for k in range(local_radius - 1):
            k_neiborhoood = []
            for node in local_region:
                k_neiborhoood += list(graph.neighbors(node))
            local_region += k_neiborhoood
            local_region = list(set(local_region))

        single_exclude_mask, single_include_mask, single_coeffs = prepare_single_coalition(coalition, local_region, num_nodes)
        set_exclude_masks += single_exclude_mask
        set_include_masks += single_include_mask
        set_coeffs += single_coeffs
        coalition_batch.append(len(single_coeffs))

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    coeffs = torch.tensor(np.stack(set_coeffs))

    marginal_contributions = \
        marginal_contribution(model, data, exclude_mask, include_mask, value_func, subgraph_build_func)

    l_shapley_values = []
    marginal_contributions = marginal_contributions.squeeze().cpu() * coeffs

    start = 0
    for c_batch in coalition_batch:
      l_shapley_values.append(marginal_contributions[start:start+c_batch].sum().item())
      start += c_batch

    return l_shapley_values


def calc_subgraphs_shapley(model, single_data: Data, subgraphs, value_func):

    l_shapley_values = l_shapley_paralell(model, subgraphs, single_data, value_func, subgraph_building_method='zero_filling')

    assert len(subgraphs) == len(l_shapley_values)

    results = dict()

    for i in range(len(subgraphs)):
      results[tuple(subgraphs[i])] = l_shapley_values[i]

    results = dict(sorted(results.items(), key=lambda item: item[1]/len(item[0]), reverse=True))

    return results