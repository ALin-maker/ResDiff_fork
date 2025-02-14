import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

from dataset import SuperResolutionDataset
from Simple_CNN import SimpleCNN
from loss import image_compare_loss


# train function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for data in tqdm(train_loader, desc="Training"):
        inputs, targets = data[:2]  # 只取 lr_image 和 hr_image
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)



# evaluate function
def calculate_psnr_batch(pred_batch, target_batch, max_val=1.0):
    """Calculate the mean PSNR of all images within a batch"""
    batch_size = pred_batch.size(0)
    mse = torch.mean(torch.pow(pred_batch - target_batch, 2), dim=[1, 2, 3])
    psnr = torch.zeros(batch_size, dtype=torch.float32, device=pred_batch.device)
    psnr[mse > 0] = 10 * torch.log10((max_val ** 2) / mse[mse > 0])
    avg_psnr = torch.mean(psnr)
    return avg_psnr.item()


def evaluate(model, dataloader, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_psnr += calculate_psnr_batch(outputs, targets)
    avg_psnr = total_psnr / len(dataloader)
    return avg_psnr


# Save the CNN prediction results
def save_res(model, dataloader, device):
    root = '/root/autodl-tmp/ResDiff/dataset/prepare_data_test_png_16_128/cnn_sr_16_128'
    model.eval()
    with torch.no_grad():
        for inputs, targets, path, _ in tqdm(dataloader, desc='Saving Results', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            image = ((outputs[0] + 1) / 2)  # Normalize image
            save_path = path[0].replace('lr_32', 'cnn_sr_16_128')
            save_image(image, save_path)


def main():
    # Setting parameters
    scale_factor = 8
    batch_size = 1
    lr = 1e-4
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hr_dir = '/root/autodl-tmp/ResDiff/dataset/prepare_data_test_png_16_128/hr_128'
    lr_dir = '/root/autodl-tmp/ResDiff/dataset/prepare_data_test_png_16_128/lr_16'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset
    train_dataset = SuperResolutionDataset(hr_dir, lr_dir, transform)
    test_dataset = SuperResolutionDataset(hr_dir, lr_dir, transform, train=False)
    
    # Set shuffle=True for better training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = SimpleCNN(scale_factor=scale_factor).to(device)

    # Check if pre-trained weights exist
    weight_path = 'pretrain_weights/cnn_weights.pth'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print(f"Loaded pre-trained weights from {weight_path}")
    else:
        print(f"No pre-trained weights found at {weight_path}. Using random initialization.")

    # Define loss function and optimizer
    criterion = image_compare_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training and evaluation
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_psnr = evaluate(model, test_loader, device)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test PSNR: {test_psnr:.4f}')

        # Save trained model
        save_model_path = 'pretrain_weights/cnn_weights.pth'
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    # Save prediction results
    save_res(model, test_loader, device)


if __name__ == '__main__':
    main()
