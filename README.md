# Predict-Toxicity-of-Molecules
The prediction of molecular properties is an important task in drug discovery. 

COSI 149 Project 3 Report
Team member: Wenhao Xie, Xianmai Liang, Zhihan Li
Description
In this project, our group experimented with various models and hyperparameters to achieve the best ROC-AUC score. The highest ROC-AUC score obtained was between 0.75 and 0.77 on the validation set.
We experimented with different models that utilize attention layers. Based on the results, we found that for the task of predicting molecular property, the use of attention layers offers limited improvement in the ROC-AUC score.
Based on the different models’ performance, we decided to use the GAT model to make predictions.
Models & Hyperparameters
We first tried to obtain the best combination of values for hyperparameters in the baseline GCN model. We tried combinations of different values for layers, batch size, learning rate, epoch and number of hidden layers. The different options we tried are given below:
Hyperparameters
Layer: 2, 3, 4 (3 default)
Batch size: 4, 8, 16,  (32 default)
Learning rate: 0.001, 0.0001. 0.0005  (0.001 default)
Epoch: 50, 100, 300, 600  (5 default)
Number of hidden layers(for GCN model): 32/64/128 powers of 2
GCN
	[Wenhao Xie]
All default max ROC AUC =0.76
[3 layers, BS=8, learning rate 0.001, epoch=100] max=0.76
[2 layers, BS=8, learning rate 0.001, epoch=100] max=0.76 but more frequently see 0.76
[2 layers, BS=8, learning rate 0.0001, epoch=100] max=0.75
[3 layers, BS=8, learning rate 0.0001, epoch=100] max=0.75 but the roc-auc for training set is still low(0.81), try larger value for epoch
[3 layers, BS=8, learning rate 0.0001, epoch=600] max=0.75
[2 layers, BS=16, learning rate 0.0001, epoch=100] max=0.75
[4 layers, BS=16, learning rate 0.0001, epoch=100, h=64] max=0.77

[Zhihan Li]
[3 layers, BS=4, learning rate 0.001, epoch=100] max= 0.77
[3 layers, BS=4, learning rate 0.0001, epoch=100] max=0.75
[3 layers, BS=4, learning rate 0.00001, epoch=100] max=0.71
[2 layers, BS=4, learning rate 0.001, epoch=100] max=0.76
[2 layers, BS=4, learning rate 0.0001, epoch=100] max=0.75


[Xianmai Liang]
[3 layers, BS=8, learning rate 0.001, epoch=100] max=0.75
[3 layers, BS=8, learning rate 0.01, epoch=100] max=0.74
[3 layers, BS=4, learning rate 0.01, epoch=100] max=0.74
[2 layers, BS=8, learning rate 0.0001, epoch=100] max=0.75
[2 layers, BS=4, learning rate 0.0001, epoch=100] max=0.75

Transformer
[Zhihan Li]
[3 layers, BS=8, learning rate 0.001, epoch=100, dropout: 0.5] max=0.75
[3 layers, BS=4, learning rate 0.001, epoch=100, dropout=0.2] max=0.77

Build Transformer from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html#torch_geometric.nn.conv.TransformerConv

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import TransformerConv, global_mean_pool as gap
from torch_geometric.nn import BatchNorm

class TransformerNet(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(TransformerNet, self).__init__()
        torch.manual_seed(42)
        self.emb = AtomEncoder(hidden_channels=32)
        self.conv1 = TransformerConv(32, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        # Node embedding
        x = self.emb(x)

        # Apply TransformerConv layers
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        # Readout layer
        x = gap(x, batch_index)  # Global Mean Pooling

        # Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        return x



GAT
The GAT model contains an attention layer, but the max roc-auc still does not exceed the baseline score from the GCN model.
	[Zhihan Li]
	[2 layers, BS=8, learning rate 0.001, epoch=100, hidden layer heads=4] max=0.75
[2 layers, BS=8, learning rate 0.001, epoch=50, hidden layer heads=6] max=0.74
[2 layers, BS=8, learning rate 0.0001, epoch=50, hidden layer heads=4] max=0.69
[3 layers, BS=8, learning rate 0.001, epoch=50, hidden layer heads=8] max=0.75
[2 layers, BS=8, learning rate 0.001, epoch=200, hidden layer heads=4] max=0.77

[Xianmai Liang]
[2 layers, BS=8, learning rate 0.001, epoch=50, hidden layer heads=4] max=0.73
[2 layers, BS=8, learning rate 0.01, epoch=100, hidden layer heads=4] max=0.72
[2 layers, BS=16, learning rate 0.001, epoch=50, hidden layer heads=4] max=0.73

[Wenhao Xie]
	[2 layers, BS=16, learning rate 0.0005, epoch=100, hidden layer heads=6] max=0.72
[2 layers, BS=16, learning rate 0.0005, epoch=100, hidden layer heads=8] max=0.74
[2 layers, BS=16, learning rate 0.0005, epoch=500, hidden layer heads=8] max=0.75

Build GAT model from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv
 from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        self.emb = AtomEncoder(hidden_channels=32)
        self.conv1 = GATConv(32, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(4*hidden_channels, hidden_channels, heads=1, concat=False)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, batch):
        x, edge_index, batch_size = batch.x, batch.edge_index, batch.batch
        x = self.emb(x)
        # 1. Obtain node embeddings
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
       # x = F.elu(self.conv2(x, edge_index))
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = gap(x, batch_size)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x




RGAT
[Zhihan Li]
	[2 layers, BS=4, learning rate 0.001, epoch=100, hidden layer heads=4] max=0.747

Build RGAT model from the given example code: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgat.py
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import RGATConv

class RGAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes, num_relations):
        super(RGAT, self).__init__()
        torch.manual_seed(42)
        self.emb = AtomEncoder(hidden_channels=32)
        self.conv1 = RGATConv(32, hidden_channels, num_relations=num_relations, concat=True)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, concat=False)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        edge_type = edge_attr.argmax(dim=-1)

        x = self.emb(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_type=edge_type))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_type=edge_type)
        x = gap(x, batch.batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

	
SuperGAT
[Zhihan Li]
	[2 layers, BS=8, learning rate 0.001, epoch=300, hidden layer heads=4] max=0.672
Build SuperGAT function from the given example code: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/super_gat.py
 class SuperGAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(SuperGAT, self).__init__()
        torch.manual_seed(42)
        self.emb = AtomEncoder(hidden_channels=32)
        self.conv1 = SuperGATConv(32, hidden_channels, heads=4, concat=True,
                                  dropout=0.5, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.conv2 = SuperGATConv(4*hidden_channels, hidden_channels, heads=1, concat=False,
                                  dropout=0.5, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.intermediate_lin = Linear(hidden_channels, 32)
        self.lin = Linear(32, num_classes)

    def forward(self, batch):
      x, edge_index, batch_size = batch.x, batch.edge_index, batch.batch
      x = self.emb(x)

      # 1. Obtain node embeddings
      x = F.dropout(x, p=0.5, training=self.training)
      x = F.elu(self.conv1(x, edge_index))
      att_loss = self.conv1.get_attention_loss()  
      x = F.dropout(x, p=0.5, training=self.training)
      x = self.conv2(x, edge_index)
      att_loss += self.conv2.get_attention_loss()

      # 2. Readout layer
      x = gap(x, batch_size)  # [batch_size, hidden_channels]

      # 3. Apply a final classifier
      x = F.dropout(x, p=0.5, training=self.training)
      x = self.lin(x)

      return x, att_loss  


Code For GAT Model,Training, Test, Load and Save
This section includes all the changes we made to the given Jupyter Notebook.

We use GAT to make predictions, and here is the GAT method we had. from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        self.emb = AtomEncoder(hidden_channels=32)
        self.conv1 = GATConv(32, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(4*hidden_channels, hidden_channels, heads=1, concat=False)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, batch):
        x, edge_index, batch_size = batch.x, batch.edge_index, batch.batch
        x = self.emb(x)
        # 1. Obtain node embeddings
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
       # x = F.elu(self.conv2(x, edge_index))
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = gap(x, batch_size)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x



We change the code on #Training to this, it will save the best model and print the best roc-auc. The path is /content/drive/MyDrive/ since we run our model on Colab.
from google.colab import drive
drive.mount('/content/drive')

We keep tracking the best model during training and use torch.save() to save the best model.

#Training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
best_val_rocauc = 0
best_model_path = "/content/drive/MyDrive/best_model.pth"
for epoch in range(1, 200):
    print("====epoch " + str(epoch))
    # Train and Evaluate
    train(model, device, train_loader, optimizer)
    train_acc = eval(model, device, train_loader)
    val_acc = eval(model, device, val_loader)
    print({'Train': train_acc, 'Validation': val_acc})
    val_rocauc = val_acc['rocauc']
    if val_rocauc > best_val_rocauc:
        best_val_rocauc = val_rocauc
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print("Saved best model with ROC-AUC: ", best_val_rocauc)

print("Finished, best ROC-AUC in Validation set is: ", best_val_rocauc)

We add a test() function to make predictions.

def test(model, device, loader, file_path):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # ignore if batch.x=None
            if batch.x is None:
                continue
            pred = model(batch)
            y_pred.extend(pred.cpu().numpy())
	   	# Save the prediction to csv
    with open(file_path, 'w', newline='') as file:
      writer = csv.writer(file)
      for pred in y_pred:
        writer.writerow(pred)


Code for Load Saved Model 
# path
test_model_path = "/content/drive/MyDrive/best_model.pth"  # path of the best model
output_file_path = "/content/drive/MyDrive/test_output.csv"  # path of the output csv

# create model and load the weight
model = GAT(32, 9, 12)
model.load_state_dict(torch.load(test_model_path))
model.to(device)


# make prediction and save it to csv
test(model, device, test_loader, output_file_path)



