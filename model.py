import torch
from torch.nn import Linear, BatchNorm1d, Dropout, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


# Baseline model
class GATEdgeAT(torch.nn.Module):
    def __init__(self, n_heads=4, edge_dim=11, num_features=30, embedding_size=128, self_attention=False,
                 multihead_attention=False, return_attention=False):
        # Init parent
        super(GATEdgeAT, self).__init__()
        torch.manual_seed(42)

        self.self_attention = self_attention
        self.multihead_attention = multihead_attention
        self.return_attention = return_attention
        self.dropout_layer = Dropout(p=0.1)

        # GAT layers
        self.initial_conv = GATConv(num_features, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        self.conv1 = GATConv(embedding_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform2 = Linear(embedding_size * n_heads, embedding_size)
        self.bn2 = BatchNorm1d(embedding_size)

        self.conv2 = GATConv(embedding_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform3 = Linear(embedding_size * n_heads, embedding_size)
        self.bn3 = BatchNorm1d(embedding_size)

        self.conv3 = GATConv(embedding_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform4 = Linear(embedding_size * n_heads, embedding_size)
        self.bn4 = BatchNorm1d(embedding_size)

        if self.self_attention:
            self.W_a = Linear(embedding_size, embedding_size, bias=False)
            self.W_b = Linear(embedding_size, embedding_size)

        if self.multihead_attention:
            self.mhat = MultiAtomAttention(embedding_size)

        # Output layer
        self.out1 = Linear(embedding_size * 2, embedding_size * 2)
        self.out2 = Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, edge_attr, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform1(hidden)
        hidden = self.bn1(hidden)

        # Other Conv layers
        # Conv layer 2
        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform2(hidden)
        hidden = self.bn2(hidden)

        # Conv layer 3
        hidden = self.conv2(hidden, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform3(hidden)
        hidden = self.bn3(hidden)

        # Conv layer 4
        hidden = self.conv3(hidden, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform4(hidden)
        hidden = self.bn4(hidden)

        if self.self_attention:
            graph_length = [0] * (batch_index[-1] + 1)
            for i in range(len(batch_index)):
                graph_length[batch_index[i]] += 1

            mol_vecs = []
            attention_weights = []
            start = 0
            for length in graph_length:
                current_hidden = torch.narrow(hidden, 0, start, length)
                att_w = torch.matmul(self.W_a(current_hidden), current_hidden.T)
                att_w = F.softmax(att_w, dim=1)
                att_hiddens = torch.matmul(att_w, current_hidden)
                att_hiddens = F.relu(self.W_b(att_hiddens))
                att_hiddens = self.dropout_layer(att_hiddens)
                mol_vec = current_hidden + att_hiddens
                mol_vecs.append(mol_vec)
                attention_weights.append(att_w)
                start += length
            hidden = torch.cat(mol_vecs, dim=0)

        if self.multihead_attention:

            graph_length = [0] * (batch_index[-1] + 1)
            for i in range(len(batch_index)):
                graph_length[batch_index[i]] += 1

            mol_vecs = []
            attention_weights = []
            start = 0
            for length in graph_length:
                current_hidden = torch.narrow(hidden, 0, start, length)
                att_hiddens, multi_att_w = self.mhat(current_hidden)
                att_hiddens = self.dropout_layer(att_hiddens)
                mol_vec = current_hidden + att_hiddens
                mol_vecs.append(mol_vec)
                attention_weights.append(multi_att_w)
                start += length
            hidden = torch.cat(mol_vecs, dim=0)

        # Pooling and concatenate
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Regression output
        out = F.relu(self.out1(hidden))
        out = self.out2(out)

        if (self.self_attention or self.multihead_attention) and self.return_attention:
            return out, hidden, attention_weights

        return out, hidden


class MultiAtomAttention(torch.nn.Module):

    def __init__(self, embedding_size):
        super(MultiAtomAttention, self).__init__()
        self.embedding_size = embedding_size
        self.dropout_layer = Dropout(p=0.2)
        self.num_heads = 4
        self.att_size = self.embedding_size // self.num_heads
        self.scale_factor = self.att_size ** -0.5

        self.W_a_q = Linear(self.embedding_size, self.num_heads * self.att_size, bias=False)
        self.W_a_k = Linear(self.embedding_size, self.num_heads * self.att_size, bias=False)
        self.W_a_v = Linear(self.embedding_size, self.num_heads * self.att_size, bias=False)
        self.W_a_o = Linear(self.num_heads * self.att_size, self.embedding_size)
        self.norm = LayerNorm(self.embedding_size, elementwise_affine=True)

    def forward(self, x):
        cur_embedding_size = x.size()

        a_q = self.W_a_q(x).view(cur_embedding_size[0], self.num_heads, self.att_size)
        a_k = self.W_a_k(x).view(cur_embedding_size[0], self.num_heads, self.att_size)
        a_v = self.W_a_v(x).view(cur_embedding_size[0], self.num_heads, self.att_size)

        a_q = a_q.transpose(0, 1)
        a_k = a_k.transpose(0, 1).transpose(1, 2)
        a_v = a_v.transpose(0, 1)

        att_a_w = torch.matmul(a_q, a_k)
        att_a_w = F.softmax(att_a_w * self.scale_factor, dim=2)
        att_a_h = torch.matmul(att_a_w, a_v)
        att_a_h = F.relu(att_a_h)
        att_a_h = self.dropout_layer(att_a_h)

        att_a_h = att_a_h.transpose(0, 1).contiguous()
        att_a_h = att_a_h.view(cur_embedding_size[0], self.num_heads * self.att_size)
        att_a_h = self.W_a_o(att_a_h)
        assert att_a_h.size() == cur_embedding_size

        att_a_h = att_a_h.unsqueeze(dim=0)
        att_a_h = self.norm(att_a_h)
        mol_vec = att_a_h.squeeze(dim=0)

        return mol_vec, torch.mean(att_a_w, axis=0)


# Fine-tuning component
class GAT_fine(torch.nn.Module):
    def __init__(self, pretrain_net):
        super(GAT_fine, self).__init__()
        torch.manual_seed(42)

        self.class_embedding_size = 32

        self.pretrain_net = pretrain_net

        # For classification
        self.linear1 = Linear(256, self.class_embedding_size)
        self.dropout = Dropout(0.2)

        self.bnL1 = BatchNorm1d(self.class_embedding_size)
        self.linear2 = Linear(self.class_embedding_size, 2)

    def forward(self, x, edge_attr, edge_index, batch_index):

        pretrain_net_outputs = self.pretrain_net(x, edge_attr, edge_index, batch_index)
        if len(pretrain_net_outputs) == 2:
            pretrain_pred, pretrain_embedding = pretrain_net_outputs
        elif len(pretrain_net_outputs) == 3:
            pretrain_pred, pretrain_embedding, _ = pretrain_net_outputs
        else:
            raise Exception("Wrong outputs from pre-trained network")

        pretrain_embedding = self.dropout(pretrain_embedding)

        # Classification head
        class_hidden = self.linear1(pretrain_embedding).relu()
        class_hidden = self.bnL1(class_hidden)
        class_hidden = self.dropout(class_hidden)

        class_out = self.linear2(class_hidden)

        return class_out, pretrain_pred
