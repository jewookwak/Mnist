import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np

class AugmentedMNIST(Dataset):
    """Augmented MNIST dataset"""

    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.augment = augment
        self.transform = self._transformations()
        self.data = self.load_data()

    def load_data(self):
        data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(self.data_dir, filename)
                label = int(filename.split('_')[-1].split('.')[0])  # Extract label from filename
                data.append((image_path, label))
        return data

    def _transformations(self):
        # Define common transformations
        transformations = [
            transforms.Grayscale(),
            transforms.Resize((32, 32))
            
        ]
        if self.augment:
            # Add augmentation transformations
            transformations.extend([
                transforms.RandomRotation(degrees=20), # 20 범위 내에서 무작위로 회전
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 좌우 및 상하로 최대 10% 이동
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),  # 무작위로 잘라서 크기 조정
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                # You can add more augmentation transforms as needed
            ])
        else:
            transformations.extend([
                  # 무작위로 잘라서 크기 조정
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                # You can add more augmentation transforms as needed
            ])
        return transforms.Compose(transformations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label
def show_image(image):
    # Convert torch tensor to NumPy array
    np_image = image.numpy()

    # Transpose the array to (H, W, C) format
    np_image = np.transpose(np_image, (1, 2, 0))

    # Display the image
    plt.imshow(np_image, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()
if __name__ == '__main__':
    # Test the implementation
    dataset = AugmentedMNIST(data_dir='C:/Users/jewoo/OneDrive/바탕 화면/mnist-classification/mnist-classification/data/train')
    print(len(dataset))  # Check total number of images
    img, label = dataset[0]  # Get the first image and label
    print(img.shape)  # Check shape of the image tensor
    print(label)  # Check label of the first image
    for i in range(10):
        img, label = dataset[i]
        print(label)
        show_image(img)
