import torch 
import torch_geometric as tg
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool, TopKPooling


class GCN(torch.nn.Module):
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, h_feats)
        self.conv3 = GCNConv(h_feats, h_feats)
        self.conv4 = GCNConv(h_feats, num_classes)
        if config["readout"] == "mean":
            self.readout = global_mean_pool
        elif config["readout"] == "sum":
            self.readout = global_add_pool
        elif config["readout"] == "max":
            self.readout = global_max_pool
        else :
            raise ValueError("pool type does not exist")
        
        self.dropout_rate = config["dropout"]
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        h = self.conv1(x, edge_index)
        h_copy = F.relu(h)
        h = F.dropout(h_copy, p=self.dropout_rate, training=self.training)
        h = self.conv2(h, edge_index)
        h_copy = F.relu(F.relu(h) + h_copy)
        h = F.dropout(h_copy, p=self.dropout_rate, training=self.training)
        h = self.conv3(h, edge_index)
        h = F.relu(F.relu(h) + h_copy)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.conv4(h, edge_index)
        
        return self.readout(h, batch=batch.batch)

class TopkGAT(torch.nn.Module):
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.dropout_rate = config["dropout"]
        
        self.conv1 = GATv2Conv(in_feats, h_feats, dropout=self.dropout_rate)
        self.pool1 = TopKPooling(h_feats, ratio=0.8)
        self.batch_norm1 = BatchNorm(h_feats)
        
        self.conv2 = GATv2Conv(h_feats, h_feats, dropout=self.dropout_rate)
        self.pool2 = TopKPooling(h_feats, ratio=0.8)
        self.batch_norm2 = BatchNorm(h_feats)
        
        self.conv3 = GATv2Conv(h_feats, h_feats, dropout=self.dropout_rate)
        self.pool3 = TopKPooling(h_feats, ratio=0.8)
        self.batch_norm3 = BatchNorm(h_feats)
        
        self.conv4 = GATv2Conv(h_feats, h_feats, dropout=self.dropout_rate)
        self.pool4 = TopKPooling(h_feats, ratio=0.8)
        self.batch_norm4 = BatchNorm(h_feats)
        
        self.dense1 = torch.nn.Linear(8*h_feats, 4*h_feats)
        self.dense2 = torch.nn.Linear(4*h_feats, num_classes)
        
        
    def forward(self, x, edge_index, batch):
        h = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        h1, edge_index, _, batch, _, _ = self.pool1(h, edge_index, batch=batch) ###Relu before 
        
        flat_1 = torch.cat([global_add_pool(h1, batch=batch), global_max_pool(h1, batch=batch)], axis=-1)
        
        h = F.relu(self.batch_norm2(self.conv2(h1, edge_index)))
        h2, edge_index, _, batch, _, _  = self.pool2(h, edge_index, batch=batch)
        
        flat_2 = torch.cat([global_add_pool(h2, batch=batch), global_max_pool(h2, batch=batch)], axis=-1)
        
        h = F.relu(self.batch_norm3(self.conv3(h2, edge_index)))
        h3, edge_index, _, batch, _, _  = self.pool3(h, edge_index, batch=batch)
        
        flat_3 = torch.cat([global_add_pool(h3, batch=batch), global_max_pool(h3, batch=batch)], axis=-1)
        
        h = F.relu(self.batch_norm4(self.conv4(h3, edge_index)))
        if torch.any(torch.isnan(self.pool4(h, edge_index, batch=batch)[0])):
            print()
        h4, edge_index, _, batch, _, _  = self.pool4(h, edge_index, batch=batch)
        
        flat_4 = torch.cat([global_add_pool(h4, batch=batch), global_max_pool(h4, batch=batch)], axis=-1)

        #flatten = flat_1 + flat_2 + flat_3 + flat_4
        flatten = torch.cat([flat_1, flat_2, flat_3, flat_4], axis=-1)
        if torch.any(torch.isnan(flatten)):
            print()
        
        h = F.relu(self.dense1(F.dropout(flatten, p=self.dropout_rate, training=self.training)))
        outputs = self.dense2(F.dropout(h, p=self.dropout_rate, training=self.training))
        
        return outputs

class TopkGCNBlock(torch.nn.Module):
    def __init__(self, in_feats: int, h_feats: int, dropout: float, ratio: float=0.8):
        super().__init__()
        self.conv = GCNConv(in_feats, h_feats, dropout=dropout)
        self.pool = TopKPooling(h_feats, ratio=ratio)
        self.batch_norm = BatchNorm(h_feats)
    
    def forward(self, x, edge_index, batch, edge_weight):
        h = F.relu(self.batch_norm(self.conv(x, edge_index, edge_weight=edge_weight)))
        h, edge_index, edge_weight, batch, _, _ = self.pool(h, edge_index, batch=batch, edge_attr=edge_weight) ###Relu before 
        
        flat_1 = torch.cat([global_mean_pool(h, batch=batch), global_max_pool(h, batch=batch)], axis=-1)
        
        return h, flat_1, edge_index, edge_weight, batch
    
class TopkGCN(torch.nn.Module):
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.dropout_rate = config["dropout"]
        self.ratio = config["topkratio"]
        self.convblock1 = TopkGCNBlock(in_feats, h_feats, dropout=self.dropout_rate, ratio=self.ratio)
        self.convblock2 = TopkGCNBlock(h_feats, h_feats, dropout=self.dropout_rate, ratio=self.ratio)
        self.convblock3 = TopkGCNBlock(h_feats, h_feats, dropout=self.dropout_rate, ratio=self.ratio)
        self.convblock4 = TopkGCNBlock(h_feats, h_feats, dropout=self.dropout_rate, ratio=self.ratio)
        self.convblock5 = TopkGCNBlock(h_feats, h_feats, dropout=self.dropout_rate, ratio=self.ratio)
        self.convblock6 = TopkGCNBlock(h_feats, h_feats, dropout=self.dropout_rate, ratio=self.ratio)
        
        self.dense1 = torch.nn.Linear(2*h_feats, 4*h_feats)
        self.dense2 = torch.nn.Linear(4*h_feats, num_classes)
        
        
    def forward(self, x, edge_index, batch, edge_weight=None):        
        h1, flat_1, edge_index, edge_weight, batch = self.convblock1(x, edge_index, batch, edge_weight=edge_weight)
        h2, flat_2, edge_index, edge_weight, batch = self.convblock2(h1, edge_index, batch, edge_weight=edge_weight)
        h3, flat_3, edge_index, edge_weight, batch = self.convblock3(h2, edge_index, batch, edge_weight=edge_weight)
        h4, flat_4, edge_index, edge_weight, batch = self.convblock4(h3, edge_index, batch, edge_weight=edge_weight)
        h5, flat_5, edge_index, edge_weight, batch = self.convblock5(h4, edge_index, batch, edge_weight=edge_weight)
        h6, flat_6, edge_index, edge_weight, batch = self.convblock6(h5, edge_index, batch, edge_weight=edge_weight)

        flatten = flat_1 + flat_2 + flat_3 + flat_4 + flat_5 + flat_6
        #flatten = torch.cat([flat_1, flat_2, flat_3, flat_4, flat_5, flat_6], axis=-1)
        if torch.any(torch.isnan(flatten)):
            print()
        
        h = F.relu(self.dense1(F.dropout(flatten, p=self.dropout_rate, training=self.training)))
        outputs = self.dense2(F.dropout(h, p=self.dropout_rate, training=self.training))
        
        return outputs

class TopkSAGEBlock(torch.nn.Module):
    def __init__(self, in_feats: int, h_feats: int, dropout: float, ratio: float=0.8):
        super().__init__()
        self.conv = SAGEConv(in_feats, h_feats, dropout=dropout)
        self.pool = TopKPooling(h_feats, ratio=ratio)
        self.batch_norm = BatchNorm(h_feats)
    
    def forward(self, x, edge_index, batch):
        h = F.relu(self.batch_norm(self.conv(x, edge_index)))
        h, edge_index, _, batch, _, _ = self.pool(h, edge_index, batch=batch) ###Relu before 
        
        flat_1 = torch.cat([global_add_pool(h, batch=batch), global_max_pool(h, batch=batch)], axis=-1)
        
        return h, flat_1, edge_index, batch
    
class TopkSAGE(torch.nn.Module):
    def __init__(self, in_feats: int, h_feats: int, num_classes:int, config=None):
        super().__init__()
        self.dropout_rate = config["dropout"]
        
        self.convblock1 = TopkSAGEBlock(in_feats, h_feats, dropout=self.dropout_rate)        
        self.convblock2 = TopkSAGEBlock(h_feats, h_feats, dropout=self.dropout_rate)       
        self.convblock3 = TopkSAGEBlock(h_feats, h_feats, dropout=self.dropout_rate)
        self.convblock4 = TopkSAGEBlock(h_feats, h_feats, dropout=self.dropout_rate)
        
        self.dense1 = torch.nn.Linear(8*h_feats, 4*h_feats)
        self.dense2 = torch.nn.Linear(4*h_feats, num_classes)
        
        
    def forward(self, x, edge_index, batch):        
        h1, flat_1, edge_index, batch = self.convblock1(x, edge_index, batch)
        h2, flat_2, edge_index, batch = self.convblock2(h1, edge_index, batch)
        h3, flat_3, edge_index, batch = self.convblock3(h2, edge_index, batch)
        h4, flat_4, edge_index, batch = self.convblock4(h3, edge_index, batch)

        #flatten = flat_1 + flat_2 + flat_3 + flat_4
        flatten = torch.cat([flat_1, flat_2, flat_3, flat_4], axis=-1)
        if torch.any(torch.isnan(flatten)):
            print()
        
        h = F.relu(self.dense1(F.dropout(flatten, p=self.dropout_rate, training=self.training)))
        outputs = self.dense2(F.dropout(h, p=self.dropout_rate, training=self.training))
        
        return outputs
        
class SimpleSignalModel(torch.nn.Module):
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