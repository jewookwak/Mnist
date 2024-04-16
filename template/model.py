import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5,self).__init__()
        self.c1 = nn.Conv2d(1,6,5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(6,16,5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(16,120,5)
        self.n1 = nn.Linear(120,84)
        self.relu = nn.ReLU()
        self.n2 = nn.Linear(84,10)

    def forward(self,x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.c3(x)
        x = self.relu(x)
        x = torch.flatten(x,1)
        x = self.n1(x)
        x = self.relu(x)
        x = self.n2(x)
        return x
        # 61,706  trainable parameters.


class LeNet5_Batch_normlization(nn.Module):
    def __init__(self):
        super(LeNet5_Batch_normlization, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """
    # def __init__(self):
    #     super(CustomMLP, self).__init__()
    #     self.fc1 = nn.Linear(32 * 32, 572)
    #     self.fc2 = nn.Linear(572, 84)
    #     self.fc3 = nn.Linear(84, 10)  # Output 10 classes

    # def forward(self, x):
    #     x = torch.flatten(x, 1)  # Flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    # 658,628 trainable parameters.
    
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 44)
        self.fc2 = nn.Linear(44, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)  # Output 10 classes

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    # 61,014 trainable parameters.

# Test the models
if __name__ == '__main__':
    lenet = LeNet5()
    custom_mlp = CustomMLP()
    print(lenet)
    print(custom_mlp)
