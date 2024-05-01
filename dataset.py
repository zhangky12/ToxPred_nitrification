import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem


class ToxDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):

        self.test = test
        self.filename = filename
        super(ToxDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):

        return self.filename

    @property
    def processed_file_names(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["Smiles"])
            try:
                f = featurizer._featurize(mol)
            except:
                print("Problematic Smiles:", row["Smiles"])
                continue

            node_feats = torch.tensor(f.node_features, dtype=torch.float)
            edge_feats = torch.tensor(f.edge_features, dtype=torch.float)
            edge_index = torch.tensor(f.edge_index, dtype=torch.long)
            label = self._get_labels(1 if row["Activity"] == "p" else 0)

            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=row["Smiles"])

            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

    def _get_labels(self, label):

        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):

        return self.data.shape[0]

    def get(self, idx):

        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data


class LogpDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(LogpDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["Smiles"])
            try:
                f = featurizer._featurize(mol)
            except:
                print("Problematic Smiles:", row["Smiles"])
                continue

            node_feats = torch.tensor(f.node_features, dtype=torch.float)
            edge_feats = torch.tensor(f.edge_features, dtype=torch.float)
            edge_index = torch.tensor(f.edge_index, dtype=torch.long)
            label = self._get_labels(row["logp"])

            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=row["Smiles"])

            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):

        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data