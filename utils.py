import torch
from sklearn.metrics import roc_auc_score
import deepchem as dc
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F


def create_single_data(smiles, label=-1):
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    mol = Chem.MolFromSmiles(smiles)
    f = featurizer._featurize(mol)

    node_feats = torch.tensor(f.node_features, dtype=torch.float)
    edge_feats = torch.tensor(f.edge_features, dtype=torch.float)
    edge_index = torch.tensor(f.edge_index, dtype=torch.long)
    label_v = torch.tensor(np.asarray([label]), dtype=torch.float)
    batch_index = torch.tensor([0] * node_feats.shape[0], dtype=torch.long)

    data = Data(x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=label_v,
                smiles=smiles,
                batch=batch_index)

    return data


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def value_func(model, data):
    with torch.no_grad():
        logits, logp = model(data.x, data.edge_index, data.edge_attr, data.batch)
        probs = F.softmax(logits, dim=-1)
        score = probs[:, 1]
    return score


def value_func_pred(model, data):
    with torch.no_grad():
        outs = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # score = F.sigmoid(pred)
    return outs[0]


def train(model, train_loader, optimizer, loss_fn):
    model.train()

    for data in train_loader:
        out, pred = model(data.x.float(), data.edge_attr.float(), data.edge_index, data.batch)
        loss = loss_fn(torch.squeeze(out), data.y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


def val(model, loader, loss_fn):
    model.eval()
    correct = 0
    total_loss = 0
    wrong_prediction = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        out, pred = model(data.x.float(), data.edge_attr.float(), data.edge_index, data.batch)
        loss = loss_fn(torch.squeeze(out), data.y)
        total_loss += loss.item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        wrong_prediction += [data.smiles[i] for i in range(len(data.y)) if pred[i] != data.y[i]]

    return correct / len(loader.dataset), total_loss / len(loader.dataset), wrong_prediction


def val_roc_auc(model, loader, loss_fn):
    model.eval()
    correct = 0
    total_loss = 0
    wrong_prediction = []

    probs = []
    trues = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        out, pred = model(data.x.float(), data.edge_attr.float(), data.edge_index, data.batch)
        loss = loss_fn(torch.squeeze(out), data.y)
        total_loss += loss.item()
        probs += list(out[:,1].detach().numpy())
        trues += list(data.y.numpy())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        wrong_prediction += [data.smiles[i] for i in range(len(data.y)) if pred[i] != data.y[i]]

    test_roc_auc = roc_auc_score(trues, probs)

    return correct / len(loader.dataset), total_loss / len(loader.dataset), test_roc_auc, wrong_prediction