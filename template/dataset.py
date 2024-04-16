# dataset.py

# Import necessary packages
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MNIST(Dataset):
    """ MNIST dataset

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((32,32))
        ])
        self.data = self.load_data()

    def load_data(self):
        data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(self.data_dir, filename)
                label = int(filename.split('_')[-1].split('.')[0])  # Extract label from filename
                data.append((image_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label

if __name__ == '__main__':
    # Test the implementation
    dataset = MNIST(data_dir='C:/Users/jewoo/OneDrive/바탕 화면/mnist-classification/mnist-classification/data/train')
    print(len(dataset))  # Check total number of images
    img, label = dataset[0]  # Get the first image and label
    print(img.shape)  # Check shape of the image tensor
    print(label)  # Check label of the first image
    print(img)
