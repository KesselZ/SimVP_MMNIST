import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from utils import *
from tqdm import tqdm
# from model.SimVP_classification import *
from model.SimVP2 import *
# from model.SimVP1 import *
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 16

# 获取第一个数据项
# 创建训练集数据
train_set = MovingMNIST( is_train=True,
                        n_frames_input=10, n_frames_output=10, num_objects=[2])
test_set = MovingMNIST( is_train=False,
                       n_frames_input=10, n_frames_output=10, num_objects=[2])

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

for img_batch, label_batch in train_dataloader:
    # 打印出批次的图像和标签的形状
    print("Shape of img batch:", img_batch.shape)
    print("Shape of label batch:", label_batch.shape)
    plot_timestep_images(img_batch)
    break  # 只获取第一个批次，然后退出循环


epochs =70
num_epoch_for_val = 100

# input is T,channel,H,W
model = SimVP_Model((10, 1, 64, 64),hid_T=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=1e-3,
                                                steps_per_epoch=len(train_dataloader),
                                                epochs=epochs)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params/1000000} M")

# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def validate_model(model, val_dataloader, device):
    model.eval()
    total_val_loss = 0.0

    val_dataloader_tqdm = tqdm(val_dataloader, desc="Validation")
    with torch.no_grad():
        for inputs, labels in val_dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs,inputs=outputs*255,inputs*255
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            val_dataloader_tqdm.set_postfix(val_loss=loss.item())

    average_val_loss = total_val_loss / len(val_dataloader)
    val_dataloader_tqdm.close()

    return average_val_loss


train_loss = 0
# Train the model
for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    train_dataloader_tqdm = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

    if os.path.exists("pause.txt"):
        pdb.set_trace()

    for i, (inputs, labels) in enumerate(train_dataloader_tqdm):
        inputs, labels = inputs.to(device), labels.to(device)

        # inputs,labels=convert_mask_to_gray_image(inputs),convert_mask_to_gray_image(labels)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss = loss

        # 正常的反向传播和优化步骤
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        train_dataloader_tqdm.set_postfix(train_loss=loss.item())

    average_train_loss = total_train_loss / len(train_dataloader)
    train_dataloader_tqdm.close()

    # 每 num_epoch_for_val 个 epoch 验证一次
    if (epoch + 1) % num_epoch_for_val == 0 or (epoch + 1) == epochs:
        average_val_loss = validate_model(model, val_dataloader, device)
        tqdm.write(
            f"Epoch Summary - Epoch [{epoch + 1}/{epochs}]: Train Loss: {average_train_loss}, Val Loss: {average_val_loss}")
    else:
        tqdm.write(
            f"Epoch Summary - Epoch [{epoch + 1}/{epochs}]: Train Loss: {average_train_loss}")
    time.sleep(0.1)
torch.save(model.state_dict(), 'SimVP2-hidT256-70.pth')
