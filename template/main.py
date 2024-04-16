import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import LeNet5, CustomMLP,LeNet5_Batch_normlization
from dataset import MNIST as CustomMNIST
from augmented_dataset import AugmentedMNIST

        
def train(model, train_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(train_loader), correct / total

def test(model, test_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(test_loader), correct / total

def plot_stats(train_stats, test_stats, title):
    plt.plot(train_stats, label='Train')
    plt.plot(test_stats, label='Test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(title + '.png')  # Save the plot as a PNG file
    # plt.show()
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available: ",torch.cuda.is_available())
    print("cuda device_count: ",torch.cuda.device_count())

    # Load datasets and create dataloaders
    train_dataset = CustomMNIST(data_dir='./data/train')
    # use augmented dataset
    Augmented_train_dataset = AugmentedMNIST(data_dir='./data/train')
    test_dataset = CustomMNIST(data_dir='./data/test')
    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=128, shuffle=True)
    Augmented_train_loader = DataLoader(Augmented_train_dataset, num_workers=2, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=128, shuffle=False)

    # Model, optimizer, and loss function
    lenet = LeNet5().to(device)
    Batch_normalization_lenet = LeNet5_Batch_normlization().to(device)
    custom_mlp = CustomMLP().to(device)
    optimizer_lenet = optim.SGD(lenet.parameters(), lr=0.01, momentum=0.9)
    optimizer_Batch_normalization_lenet = optim.SGD(Batch_normalization_lenet.parameters(), lr=0.01, momentum=0.9)
    optimizer_custom_mlp = optim.SGD(custom_mlp.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 50
    lenet_train_loss_history, lenet_train_acc_history = [], []
    Batch_normalization_lenet_train_loss_history, Batch_normalization_lenet_train_acc_history = [], []
    custom_mlp_train_loss_history, custom_mlp_train_acc_history = [], []
    lenet_test_loss_history, lenet_test_acc_history = [], []
    Batch_normalization_lenet_test_loss_history, Batch_normalization_lenet_test_acc_history = [], []
    custom_mlp_test_loss_history, custom_mlp_test_acc_history = [], []
    for epoch in range(num_epochs):
        lenet_train_loss, lenet_train_acc = train(lenet, train_loader, device, criterion, optimizer_lenet)
        Batch_normalization_lenet_train_loss, Batch_normalization_lenet_train_acc = train(Batch_normalization_lenet, Augmented_train_loader, device, criterion, optimizer_Batch_normalization_lenet)
        custom_mlp_train_loss, custom_mlp_train_acc = train(custom_mlp, train_loader, device, criterion, optimizer_custom_mlp)
        lenet_test_loss, lenet_test_acc = test(lenet, test_loader, device, criterion)
        Batch_normalization_lenet_test_loss, Batch_normalization_lenet_test_acc = test(Batch_normalization_lenet, test_loader, device, criterion)
        custom_mlp_test_loss, custom_mlp_test_acc = test(custom_mlp, test_loader, device, criterion)

        lenet_train_loss_history.append(lenet_train_loss)
        lenet_train_acc_history.append(lenet_train_acc)
        Batch_normalization_lenet_train_loss_history.append(Batch_normalization_lenet_train_loss)
        Batch_normalization_lenet_train_acc_history.append(Batch_normalization_lenet_train_acc)
        custom_mlp_train_loss_history.append(custom_mlp_train_loss)
        custom_mlp_train_acc_history.append(custom_mlp_train_acc)
        lenet_test_loss_history.append(lenet_test_loss)
        lenet_test_acc_history.append(lenet_test_acc)
        Batch_normalization_lenet_test_loss_history.append(Batch_normalization_lenet_test_loss)
        Batch_normalization_lenet_test_acc_history.append(Batch_normalization_lenet_test_acc)
        custom_mlp_test_loss_history.append(custom_mlp_test_loss)
        custom_mlp_test_acc_history.append(custom_mlp_test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"LeNet-5 Train Loss: {lenet_train_loss:.4f}, Train Acc: {lenet_train_acc:.4f}, "
              f"Test Loss: {lenet_test_loss:.4f}, Test Acc: {lenet_test_acc:.4f}"
              )
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Custom MLP Train Loss: {custom_mlp_train_loss:.4f}, Train Acc: {custom_mlp_train_acc:.4f}, "
              f"Test Loss: {custom_mlp_test_loss:.4f}, Test Acc: {custom_mlp_test_acc:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Batch_normalization_data_Augmentation_LeNet-5 Train Loss: {Batch_normalization_lenet_train_loss:.4f}, Train Acc: {Batch_normalization_lenet_train_acc:.4f}, "
              f"Test Loss: {Batch_normalization_lenet_test_loss:.4f}, Test Acc: {Batch_normalization_lenet_test_acc:.4f}"
              )

    # Plot training and testing statistics
    plot_stats(lenet_train_loss_history, lenet_test_loss_history, 'Loss (LeNet-5)')
    plot_stats(lenet_train_acc_history, lenet_test_acc_history, 'Accuracy (LeNet-5)')
    plot_stats(Batch_normalization_lenet_train_loss_history, Batch_normalization_lenet_test_loss_history, 'Loss (Batch Normalization Data Augmentation LeNet-5)')
    plot_stats(Batch_normalization_lenet_train_acc_history, Batch_normalization_lenet_test_acc_history, 'Accuracy (Batch Normalization Data Augmentation LeNet-5)')
    plot_stats(custom_mlp_train_loss_history, custom_mlp_test_loss_history, 'Loss (Custom MLP)')
    plot_stats(custom_mlp_train_acc_history, custom_mlp_test_acc_history, 'Accuracy (Custom MLP)')

if __name__ == '__main__':
    main()
