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
# 1. Load and split the dataset

lr, num_epochs, batch_size = 0.0001, 200, 16
data = scio.loadmat('./DATASET_shice_NV++.mat')

# Input features
feature = torch.tensor(data['feature'], dtype=torch.float32)  # 234, 25, 5453
feature = feature.permute(2, 1, 0).unsqueeze(1)  # 5453, 25, 234

# Output labels
label = torch.tensor(data['label'], dtype=torch.float32)  # 2, 5453
label = label.permute(1, 0)  # 5453, 2

# Create a dataset object (not strictly necessary, but kept for clarity)
dataset = torch.utils.data.TensorDataset(feature, label)

# Randomly shuffle indices
shuffled_indices = torch.randperm(len(dataset))
shuffled_feature = feature[shuffled_indices]
shuffled_label = label[shuffled_indices]

# Split dataset into training and validation sets (e.g., 80% / 20%)
train_ratio = 0.2
split_idx = int(train_ratio * len(shuffled_feature))
train_feature, val_feature = shuffled_feature[:split_idx], shuffled_feature[split_idx:]
train_label, val_label = shuffled_label[:split_idx], shuffled_label[split_idx:]

# Create DataLoader for training and validation sets
train_dataset = torch.utils.data.TensorDataset(train_feature, train_label)
train_iter = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_dataset = torch.utils.data.TensorDataset(val_feature, val_label)
test_iter = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=len(val_dataset),  # Use the entire validation set in one pass
    shuffle=False
)

##########
# 2. Initialize model parameters (used in train_ch6)

##########
# 3. Define the network architecture

# Overall network definition
net = nn.Sequential(
    ResidualBlock(1, 8),                     # Initial residual block
    nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
    DeformConv2d_l(8, 32),
    nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)),
    DeformConv2d_l(32, 16),
    nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)),
    View(),                                  # [batch, sequence length, embedding dimension]
    SelfAttention(8, 16, 16, 0.2),
    # (num_attention_heads, input_size, hidden_size, hidden_dropout_prob)
    Anti_Embad(16),
    nn.Dropout(p=0.3),
    nn.Linear(50, 2),
    nn.Softmax(dim=1)                        # Output probability distribution
)

net.load_state_dict(torch.load('DCNN-SA-PARA.pth'))

def Conv_block(input_channels, num_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(num_channels),
        nn.ReLU()
    )

# Check model output shape to ensure consistency with expectations
X = torch.rand(size=(1, 1, 25, 234), dtype=torch.float32)
print(X.size())
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

##########
# 4. Train the model

def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """Compute model accuracy on a dataset using GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device

    # Accumulate correct predictions and total samples
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT-style fine-tuning (not used here)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.shape[0])

    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train the model on GPU (as defined in Chapter 6)."""
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
        # Sum of training loss, training accuracy, and number of samples
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

        print(f'epoch {epoch + 1:.0f}, loss {train_l:.11f}, '
              f'train acc {train_acc:.11f}, test acc {test_acc:.11f}')

        losslist.append(train_l)
        testacclist.append(test_acc)
        trainacclist.append(train_acc)

        # Compute four evaluation metrics
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

        # Compute accuracy, precision, recall, and F1-score
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

    # Save training statistics
    pd.DataFrame(losslist).to_excel('loss_try_qianyi_NOSD.xlsx', header='loss', index=True)
    pd.DataFrame(testacclist).to_excel('test_acc_try_qianyi_NOSD.xlsx', header='test_acc', index=True)
    pd.DataFrame(trainacclist).to_excel('train_acc_try_qianyi_NOSD.xlsx', header='train_acc', index=True)
    pd.DataFrame(fourlist).to_excel('four_try_qianyi_NOSD.xlsx', header='four', index=True)

    # Plot training curves
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(num=1, figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("epoch", fontsize=30)
    ax.set_xlim([1, num_epochs])
    ax.plot(torch.arange(num_epochs).numpy(), losslist, "b-", label="Loss")
    ax.plot(torch.arange(num_epochs).numpy(), trainacclist, "r-", label="Training Accuracy")
    ax.plot(torch.arange(num_epochs).numpy(), testacclist, "g-", label="Test Accuracy")
    plt.tick_params(labelsize=30)
    ax.legend(loc=1, labelspacing=1, handlelength=3, fontsize=30)
    plt.savefig("test_new.svg", dpi=600, format="svg")
    plt.show()


def accuracy(y_hat, y):
    """Return the number of correctly classified samples."""
    predicted_labels = torch.argmax(y_hat, dim=1)
    true_labels = torch.argmax(y, dim=1)
    correct = (predicted_labels == true_labels).sum().item()
    return correct


def calculate_metrics(conf_matrix):
    """
    Compute accuracy, precision, recall, and F1-score.
    :param conf_matrix: Confusion matrix of shape (num_classes, num_classes)
    """
    TP = conf_matrix[0, 0].item()  # True positive
    FP = conf_matrix[1, 0].item()  # False positive
    TN = conf_matrix[1, 1].item()  # True negative
    FN = conf_matrix[0, 1].item()  # False negative

    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


def calculate_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    Compute the confusion matrix.
    :param y_true: True labels (one-hot encoded), shape (n_samples, num_classes)
    :param y_pred: Model outputs (probability distributions), shape (n_samples, num_classes)
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
# torch.save(net.state_dict(), 'DCNN-SA-PARA_qianyi.pth')
