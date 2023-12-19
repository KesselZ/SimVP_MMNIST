import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from utils import *
from model.SimVP2 import *
# from model.SimVP1 import *
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim = 1
batch_size = 4

train_dataset = MMNISTDataset('train')
val_dataset = MMNISTDataset('val')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = SimVP_Model((10, 1, 64, 64)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
saved_state_dict = torch.load('weight/SimVP2-default.pth', map_location=device)
model.load_state_dict(saved_state_dict)

# 11to11
with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_dataloader):
        plot_timestep_images(inputs)
        plot_timestep_images(labels)

        model.eval()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        plot_timestep_images(outputs.cpu().numpy())

        mean_ssim = calculate_mean_ssim(outputs, labels)
        print(f"Average SSIM: {mean_ssim}")

        mse_loss = MSE(outputs, labels)
        print(f"MSE Loss: {mse_loss.item()}")
        break

def validate_model(model, val_dataloader, device, criterion=F.mse_loss):
    model.eval()
    total_val_loss = 0.0
    total_ssim = 0.0

    val_dataloader_tqdm = tqdm(val_dataloader, desc="Validation")
    with torch.no_grad():
        for inputs, labels in val_dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            mse_loss = MSE(outputs, labels)
            ssim = calculate_mean_ssim(outputs, labels)
            total_val_loss += mse_loss.item()
            total_ssim += ssim

            val_dataloader_tqdm.set_postfix(val_loss=mse_loss.item(), ssim=ssim)

    average_val_loss = total_val_loss / len(val_dataloader)
    average_ssim = total_ssim / len(val_dataloader)
    val_dataloader_tqdm.close()

    return average_val_loss, average_ssim

print(validate_model(model, val_dataloader, device))