import torch
from torch import nn
import scipy.io as scio
from d2l import torch as d2l
import os
import matplotlib.pyplot as plt
from DeformConv2d_l import DeformConv2d_l
from SelfAttention import SelfAttention
import numpy as np
import pandas as pd
from View import View
from Anti_Embad import Anti_Embad
import torch.utils.data as data
from ResidualBlock import ResidualBlock

##########
# 1. Load dataset
lr, num_epochs, batch_size = 0.001, 500, 16
data = scio.loadmat('./DATASET2_0-1.mat')

# Input features
feature = torch.tensor(data['feature'], dtype=torch.float32)  # 234, 25, 5453
feature = feature.permute(2, 1, 0).unsqueeze(1)                # 5453, 1, 25, 234

# Output labels
label = torch.tensor(data['label'], dtype=torch.float32)       # 2, 5453
label = label.permute(1, 0)                                    # 5453, 2

# --------------------------------------------------
#  Load speed condition labels (KEY POINT)
# --------------------------------------------------
# speed_id: shape (5453,), values in {0,1,2,3,4}
speed_id = torch.tensor(data['speed_id'].squeeze(), dtype=torch.long)

# --------------------------------------------------
#  Split by speed conditions
#    train: 3 speeds, val: 1 speed, test: 1 speed
# --------------------------------------------------
train_speeds = [0, 1, 2]
val_speeds   = [3]
test_speeds  = [4]

train_mask = torch.isin(speed_id, torch.tensor(train_speeds))
val_mask   = torch.isin(speed_id, torch.tensor(val_speeds))
test_mask  = torch.isin(speed_id, torch.tensor(test_speeds))

train_feature = feature[train_mask]
train_label   = label[train_mask]

val_feature = feature[val_mask]
val_label   = label[val_mask]

test_feature = feature[test_mask]
test_label   = label[test_mask]

# --------------------------------------------------
# Shuffle ONLY training set
# --------------------------------------------------
perm = torch.randperm(len(train_feature))
train_feature = train_feature[perm]
train_label   = train_label[perm]

# --------------------------------------------------
#  DataLoaders
# --------------------------------------------------
train_dataset = torch.utils.data.TensorDataset(train_feature, train_label)
train_iter = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_dataset = torch.utils.data.TensorDataset(val_feature, val_label)
val_iter = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=len(val_dataset),
    shuffle=False
)

test_dataset = torch.utils.data.TensorDataset(test_feature, test_label)
test_iter = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=len(test_dataset),
    shuffle=False
)

##########
# 2. Initialize model parameters (functions defined in train_ch6)
##########

##########
# 3. Define the network
##########

# Construct the overall network
net = nn.Sequential(
    ResidualBlock(1, 8),  # Main branch: 5×32 → 1×32
    nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),  # 3×12
    DeformConv2d_l(8, 32),
    nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)),  # 3×12
    DeformConv2d_l(32, 16),
    nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)),  # 3×12
    View(),  # [batch size, sequence length, embedding dimension]
    SelfAttention(8, 16, 16, 0.2),
    # (num_attention_heads, input_size, hidden_size, hidden_dropout_prob)
    Anti_Embad(16),
    nn.Dropout(p=0.3),
    nn.Linear(50, 2),
    nn.Softmax(dim=1)  # Output probability distribution
)

'''
def Conv_block(input_channels, num_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(num_channels),
        nn.ReLU()
    )
'''

# Check model output size to ensure consistency with expectations
X = torch.rand(size=(1, 1, 25, 234), dtype=torch.float32)
print(X.size())
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

##########
# 4. Train the model
##########

# Training function (loss function included)
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute model accuracy on a dataset using GPU."""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    # Number of correct predictions and total predictions
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT fine-tuning (introduced later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.shape[0])
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train the model using GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)

    losslist = []
    testacclist = []
    trainacclist = []
    fourlist = []

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, and number of samples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch + 1:.0f}, loss {train_l:.11f}, train acc {train_acc:.11f}, '
              f'test acc {test_acc:.11f}')  # Metrics per epoch

        losslist.append(train_l)
        testacclist.append(test_acc)
        trainacclist.append(train_acc)

        # Four key performance metrics
        net.eval()
        all_preds = torch.tensor([], dtype=torch.float32)   # Store all predicted probabilities
        all_labels = torch.tensor([], dtype=torch.float32)  # Store all true labels (one-hot)

        with torch.no_grad():
            for inputs, labels in test_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                all_preds = torch.cat([all_preds, outputs.cpu()], dim=0)
                all_labels = torch.cat([all_labels, labels.cpu()], dim=0)

        # Compute confusion matrix
        conf_matrix = calculate_confusion_matrix(all_labels, all_preds)

        # Compute four evaluation metrics
        accuracy_, precision, recall, f1_score = calculate_metrics(conf_matrix)
        fourlist.append([accuracy_, precision, recall, f1_score])

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Accuracy: {accuracy_:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    print(f'loss {train_l:.11f}, train acc {train_acc:.11f}, '
          f'test acc {test_acc:.11f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

    # Save training results
    pd.DataFrame(losslist).to_excel('loss_try_NODS_1.xlsx', header='loss', index=True)
    pd.DataFrame(testacclist).to_excel('test_acc_try_NODS_1.xlsx', header='test_acc', index=True)
    pd.DataFrame(trainacclist).to_excel('train_acc_try_NODS_1.xlsx', header='train_acc', index=True)
    pd.DataFrame(fourlist).to_excel('four_try_NODS_1.xlsx', header='four', index=True)

    # Plot training curves
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(num=1, figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("epoch", fontsize=30)
    ax.set_xlim([1, num_epochs])
    ax.plot(torch.arange(num_epochs).numpy(), losslist, label="Loss")
    ax.plot(torch.arange(num_epochs).numpy(), trainacclist, label="Training accuracy")
    ax.plot(torch.arange(num_epochs).numpy(), testacclist, label="Test accuracy")
    plt.tick_params(labelsize=30)
    ax.legend(loc=1, fontsize=30)
    plt.savefig("test_new.svg", dpi=600, format="svg")
    plt.show()

def accuracy(y_hat, y):
    # y_hat: model output; y: ground-truth labels
    predicted_labels = torch.argmax(y_hat, dim=1)
    true_labels = torch.argmax(y, dim=1)
    correct = (predicted_labels == true_labels).sum().item()
    return correct  # Return the number of correctly classified samples

def calculate_metrics(conf_matrix):
    """
    Compute four evaluation metrics.
    :param conf_matrix: Confusion matrix with shape (num_classes, num_classes)
    :return: accuracy, precision, recall, f1_score
    """
    TP = conf_matrix[0, 0].item()  # True positives
    FP = conf_matrix[1, 0].item()  # False positives
    TN = conf_matrix[1, 1].item()  # True negatives
    FN = conf_matrix[0, 1].item()  # False negatives

    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def calculate_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    Compute the confusion matrix.
    :param y_true: Ground-truth labels with shape (n_samples, num_classes), one-hot encoded
    :param y_pred: Model outputs with shape (n_samples, num_classes), probability distributions
    :param num_classes: Number of classes
    :return: Confusion matrix with shape (num_classes, num_classes)
    """
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    true_labels = torch.argmax(y_true, dim=1)
    pred_labels = torch.argmax(y_pred, dim=1)

    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t, p] += 1

    return conf_matrix


train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# torch.save(net.state_dict(), 'DCNN-SA-PARA.pth')
