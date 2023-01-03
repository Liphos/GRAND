"""Definining the model class"""

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GatedGraphConv
import torch_geometric.nn as tgn

class DenseNet(torch.nn.Module):
    """MLP model"""
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.dropout_rate = config["dropout"]
        self.num_layers = config["num_layers"]-2
        self.dense_in = torch.nn.Linear(in_feats, h_feats)
        self.dense = torch.nn.ModuleList([torch.nn.Linear(h_feats, h_feats) for _ in range(self.num_layers)])
        self.dense_out = torch.nn.Linear(h_feats, num_classes)


    def forward(self, inputs, edge_index, batch, edge_weight=None):
        """equivalent to __call__"""
        flatten = tgn.pool.global_max_pool(inputs, batch=batch)
        #flatten = torch.cat([tgn.pool.global_mean_pool(inputs, batch=batch),
        #                     tgn.pool.global_max_pool(inputs, batch=batch)], axis=-1)

        if torch.any(torch.isnan(flatten)):
            print("Nan detected in the prediction")

        h_emb = F.relu(self.dense_in(flatten))
        for layer in self.dense:
            h_emb = F.relu(layer(F.dropout(h_emb, p=self.dropout_rate)))
        outputs = self.dense_out(F.dropout(h_emb, p=self.dropout_rate))

        return outputs

class GCNBlock(torch.nn.Module):
    """Basic GCN Block"""
    def __init__(self, in_feats: int, h_feats: int, add_residue: bool=False):
        super().__init__()
        self.conv = GCNConv(in_feats, h_feats)
        self.batch_norm = tgn.norm.GraphNorm(h_feats)
        self.add_residue = add_residue

    def forward(self, inputs, edge_index, batch, edge_weight):
        """equivalent to __call__"""
        h_emb = self.batch_norm(self.conv(inputs, edge_index, edge_weight=edge_weight))
        if self.add_residue:
            h_emb = F.relu(h_emb + inputs)
        else:
            h_emb = F.relu(h_emb)

        flat = tgn.pool.global_max_pool(h_emb, batch=batch)
        #flat = torch.cat([tgn.pool.global_mean_pool(h_emb, batch=batch),
        #                  tgn.pool.global_max_pool(h_emb, batch=batch)], axis=-1)

        return h_emb, flat, edge_index, edge_weight, batch

class GCN(torch.nn.Module):
    """Basic GCN model"""
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.dropout_rate = config["dropout"]
        self.num_layers = config["num_layers"] - 1
        self.convinput = GCNBlock(in_feats, h_feats)
        self.convblocks = torch.nn.ModuleList([GCNBlock(h_feats, h_feats, add_residue=True) for _ in range(self.num_layers)])

        self.dense1 = torch.nn.Linear(h_feats, 4*h_feats)
        self.dense2 = torch.nn.Linear(4*h_feats, num_classes)


    def forward(self, inputs, edge_index, batch, edge_weight=None):
        """equivalent to __call__"""
        h_emb, flat_1, edge_index, edge_weight, batch = self.convinput(inputs,
                                                                       edge_index,
                                                                       batch,
                                                                       edge_weight=edge_weight)
        flatten = flat_1
        for i in range(self.num_layers):
            h_emb, flat, edge_index, edge_weight, batch = self.convblocks[i](h_emb,
                                                                             edge_index,
                                                                             batch,
                                                                             edge_weight=edge_weight)
            flatten += flat

        if torch.any(torch.isnan(flatten)):
            print("Nan detected in the prediction")

        h_emb = F.relu(self.dense1(F.dropout(flatten, p=self.dropout_rate)))
        outputs = self.dense2(F.dropout(h_emb, p=self.dropout_rate))

        return outputs

class TopkGCNBlock(torch.nn.Module):
    """TopkGCN base block"""
    def __init__(self, in_feats: int, h_feats: int, ratio: float=0.8, add_residue=False):
        super().__init__()
        self.conv = GCNConv(in_feats, h_feats)
        self.pool = tgn.pool.TopKPooling(h_feats, ratio=ratio)
        self.batch_norm = tgn.norm.BatchNorm(h_feats)
        self.add_residue = add_residue

    def forward(self, inputs, edge_index, batch, edge_weight):
        """equivalent to __call__"""
        h_emb = self.batch_norm(self.conv(inputs, edge_index, edge_weight=edge_weight))
        if self.add_residue:
            h_emb = F.relu(h_emb + inputs)
        else:
            h_emb = F.relu(h_emb)
        h_emb, edge_index, edge_weight, batch, _, _ = self.pool(h_emb,
                                                                edge_index,
                                                                batch=batch,
                                                                edge_attr=edge_weight)
        ###Relu before the pooling layer

        flat = torch.cat([tgn.pool.global_mean_pool(h_emb, batch=batch),
                          tgn.pool.global_max_pool(h_emb, batch=batch)],
                          axis=-1)

        return h_emb, flat, edge_index, edge_weight, batch

class TopkGCN(torch.nn.Module):
    """GCN Hierarchical architecture with topk pooling"""
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.dropout_rate = config["dropout"]
        self.ratio = config["topkratio"]
        self.num_layers = config["num_layers"] - 1
        self.convinput = TopkGCNBlock(in_feats, h_feats, ratio=self.ratio)
        self.convblocks = torch.nn.ModuleList([TopkGCNBlock(h_feats, h_feats, ratio=self.ratio) for _ in range(self.num_layers)])

        self.dense1 = torch.nn.Linear(2*h_feats, 4*h_feats)
        self.dense2 = torch.nn.Linear(4*h_feats, num_classes)


    def forward(self, inputs, edge_index, batch, edge_weight=None):
        """equivalent to __call__"""
        h_emb, flat_1, edge_index, edge_weight, batch = self.convinput(inputs,
                                                                       edge_index,
                                                                       batch,
                                                                       edge_weight=edge_weight)
        flatten = flat_1
        for i in range(self.num_layers):
            h_emb, flat, edge_index, edge_weight, batch = self.convblocks[i](h_emb,
                                                                             edge_index,
                                                                             batch,
                                                                             edge_weight=edge_weight)
            flatten += flat

        #flatten = torch.cat([flat_1, flat_2, flat_3, flat_4], axis=-1)
        if torch.any(torch.isnan(flatten)):
            print("Nan detected in the prediction")

        h_emb = F.relu(self.dense1(F.dropout(flatten, p=self.dropout_rate)))
        outputs = self.dense2(F.dropout(h_emb, p=self.dropout_rate))

        return outputs


class GatedGCN(torch.nn.Module):
    """Implementation of the GatedGCN"""
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.dropout_rate = config["dropout"]
        self.conv = GatedGraphConv(h_feats, config["num_layers"])

        self.dense1 = torch.nn.Linear(h_feats, 4*h_feats)
        self.dense2 = torch.nn.Linear(4*h_feats, num_classes)


    def forward(self, inputs, edge_index, batch, edge_weight=None):
        """equivalent to __call__"""
        h_emb = self.conv(inputs, edge_index, edge_weight=edge_weight)

        flatten = torch.cat([tgn.pool.global_max_pool(h_emb, batch=batch)], axis=-1)
        #flatten = torch.cat([flat_1, flat_2, flat_3, flat_4], axis=-1)
        if torch.any(torch.isnan(flatten)):
            print("Nan detected in the prediction")

        h_emb = F.relu(self.dense1(F.dropout(flatten, p=self.dropout_rate)))
        outputs = self.dense2(F.dropout(h_emb, p=self.dropout_rate))

        return outputs

class SimpleSignalModel(torch.nn.Module):
    """Model containing a cnn network to work directly on the signals"""
    def __init__(self, last_activation:str=None):
        super(SimpleSignalModel, self).__init__()
        self.layers = []

        self.conv1 = torch.nn.Conv1d(3, 32, kernel_size=15, padding=7)
        self.layers.append(self.conv1)

        self.batch_norm1 = torch.nn.BatchNorm1d(64)
        self.layers.append(self.batch_norm1)

        self.conv2 = torch.nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.layers.append(self.conv2)

        self.batch_norm2 = torch.nn.BatchNorm1d(64)
        self.layers.append(self.batch_norm2)

        self.conv3 = torch.nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.layers.append(self.conv3)

        self.batch_norm3 = torch.nn.BatchNorm1d(32)
        self.layers.append(self.batch_norm3)

        self.conv4 = torch.nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.layers.append(self.conv3)

        self.batch_norm4 = torch.nn.BatchNorm1d(32)
        self.layers.append(self.batch_norm4)

        self.conv5 = torch.nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.layers.append(self.conv3)

        self.batch_norm5 = torch.nn.BatchNorm1d(32)
        self.layers.append(self.batch_norm4)

        self.dense1 = torch.nn.Linear(12*32, 512)
        self.layers.append(self.dense1)
        self.dense2 = torch.nn.Linear(512, 1)
        self.layers.append(self.dense2)

        self.dropout = torch.nn.Dropout(0.1)
        self.layers.append(self.dropout)

        self.activation = torch.nn.ReLU()
        self.layers.append(self.activation)

        self.maxpool = torch.nn.MaxPool1d(15, stride=4, padding=7)
        self.layers.append(self.maxpool)

        self.flatten = torch.nn.Flatten()
        if last_activation == "Sigmoid":
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = torch.nn.Identity()

    def forward(self, x):
        """equivalent to __call__"""
        x = self.maxpool(self.conv1(x))
        x = self.activation(self.batch_norm1(x))

        x = self.activation(self.batch_norm2(self.conv2(x)))

        x = self.activation(self.activation(self.batch_norm3(self.conv3(x))) + x)
        x = self.maxpool(x)

        x = self.activation(self.activation(self.batch_norm4(self.conv4(x))) + x)
        x = self.maxpool(x)

        x = self.activation(self.activation(self.batch_norm5(self.conv5(x))) + x)
        #x = self.maxpool(x)

        x = self.flatten(x)
        """
        x = self.activation(self.dense1(x))
        x = self.last_activation(self.dense2(x))
        """
        return x

    def save_txt(self, filename:str):
        """Save the layers in a txt
        Args:
            filename (str): path to the txt file
        """
        with open(filename, 'w') as f:
            for layer in self.layers:
                f.write(str(layer._get_name) + "\n")
        f.close()


def algorithm_from_name(name:str):
    """Returns algorithm given the name of the algorithm"""
    if name in _dict_from_name:
        return _dict_from_name[name]
    else:
        raise ValueError(f"This model {name} does not exist")

_dict_from_name = {
    "GCN": GCN,
    "TopkGCN": TopkGCN,
    "GatedGCN": GatedGCN,
    "Dense": DenseNet,
}
