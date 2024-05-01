from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors


def visualize_attention(smiles, attention_weight):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = len(mol.GetAtoms())
    atomSum_weights = attention_weight.sum(axis=0).cpu().numpy()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[0, 0.4, 0.8], [1, 1, 1], [0.97, 0.46, 0.43]])

    Amean_weight = atomSum_weights / num_atoms
    nanMean = np.nanmean(Amean_weight)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, Amean_weight - nanMean,
                                                     alpha=0.3,
                                                     size=(150, 150), colorMap=cmap)
    plt.show()
    plt.close(fig)


def convert_att(att, n_atom):
    """
    Convert the attention to a matrix, while each row representing an atom, and each column representing a
    contribution. For example, the location [0,1] represents the contribution of atom 1 to atom 0
    :param att: attention weights from gat_conv_layer
    :param n_atom: number of atoms in the molecule
    :return: contribution matrix
    """

    att_avg = torch.mean(att[1], axis=1)

    converted_att = torch.zeros((n_atom, n_atom))

    for i in range(len(att[0][0])):
        converted_att[att[0][1][i]][att[0][0][i]] += att_avg[i]

    return converted_att


def calculate_gat_att(x, edge_index, edge_attr, gat_conv_layer, transform_layer, batch_norm_layer, n_atom):
    """
    calculate the attention matrix for a GATConv layer, and keep track of the input (hidden) to the next
    convolution layer
    :param x: input of the convolution layer
    :param edge_index
    :param edge_attr
    :param gat_conv_layer
    :param transform_layer
    :param batch_norm_layer
    :param n_atom
    :return: hidden to the next convolution layer, contribution matrix
    """

    forward_res = gat_conv_layer.forward(x, edge_index, edge_attr, return_attention_weights=True)
    att = convert_att(forward_res[1], n_atom)
    hidden = forward_res[0]
    hidden = F.tanh(hidden)
    hidden = transform_layer(hidden)
    hidden = batch_norm_layer(hidden)

    return hidden, att


def extract_layers(model):
    """
    Extract GATConv layers, head transform layers (transform dimensions back to the embedding size), and batch
    normalization layers.
    :param model
    :return: convolution layers, transform layers, and normalization layers
    """

    conv_layers = [model.initial_conv, model.conv1, model.conv2, model.conv3]
    transform_layers = [model.head_transform1, model.head_transform2, model.head_transform3, model.head_transform4]
    norm_layers = [model.bn1, model.bn2, model.bn3, model.bn4]

    return conv_layers, transform_layers, norm_layers


def extract_multi_head_attention(x, edge_index, edge_attr, batch, model):
    # Calculate the multi-head attention
    # Only accept one sample at a time

    layer_att = model(x, edge_index, edge_attr, batch)[2][0].detach()
    return layer_att


def calculate_overall_att(x, edge_index, edge_attr, batch, model, n_atom):
    # Calculate the corrected attentions of inputs with a model.
    # Only accept one sample at a time

    conv_layers, transform_layers, norm_layers = extract_layers(model)

    assert len(conv_layers) == len(transform_layers) and len(transform_layers) == len(norm_layers)

    n_layers = len(conv_layers)
    atts = []
    hidden = x

    for i in range(n_layers):
        hidden, att = calculate_gat_att(hidden, edge_index, edge_attr, conv_layers[i], transform_layers[i],
                                        norm_layers[i], n_atom)
        atts.append(att)

    # The attention (distribution) of atoms before multi-head attention layer.
    # In the method section - Attention Correction of paper: C1 x C2 x C3 x C4
    pre_att = atts[0]
    for i in range(1, n_layers):
        pre_att = torch.matmul(atts[i], pre_att)

    # The attention from multi-head attention layer.
    # The next two lines should be equivalent.

    # layer_att = extract_multi_head_attention(x, edge_index, edge_attr, batch, model)
    layer_att = model(x, edge_index, edge_attr, batch)[2][0]

    # The attention after multiplying multi-head attention with embeddings (hidden)
    post_att = torch.round(torch.matmul(layer_att, pre_att))

    # The attention is added back to embeddings, so the overall attention is
    overall_att = pre_att + post_att
    overall_att = overall_att.detach()

    # Normalize for each atom
    overall_att = overall_att / overall_att.sum(dim=-1).unsqueeze(-1)

    return overall_att
