import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as cal_ssim
import torch.utils.data as data
import gzip
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def plot_images(tensor):
    """
    绘制一个图像张量，每行最多四张图像。
    参数:
        tensor (torch.Tensor): 一个形状为 [B, C, H, W] 的图像张量，其中C为3。
    """
    # 确保tensor是numpy数组
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # 调整数据范围到[0, 1]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()

    tensor = np.transpose(tensor, (0, 2, 3, 1))  # 重排形状为[B, H, W, C]

    # 计算总行数
    num_images = tensor.shape[0]
    num_rows = np.ceil(num_images / 4).astype(int)

    # 设置绘图
    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 3 * num_rows))
    axes = axes.flatten()

    # 绘制每张图像
    for i in range(num_images):
        img = tensor[i]
        axes[i].imshow(img)
        axes[i].axis('off')  # 关闭坐标轴

    # 隐藏剩余的坐标轴（如果有的话）
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_timestep_images(tensor):
    """
    绘制一个图像张量，每个批次的图像按时间步展开，每行最多四张子图像。
    参数:
        tensor (torch.Tensor): 一个形状为 [B, T, C, H, W] 的图像张量，其中C可以是1或3。
    """
    # 确保tensor是numpy数组
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # 调整数据范围到[0, 1]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()

    # 获取批次大小、时间步数、通道数
    B, T, C, H, W = tensor.shape

    # 如果是灰度图像，重复通道以适应imshow
    if C == 1:
        tensor = np.repeat(tensor, 3, axis=2)

    # 重排形状为 [B, T, H, W, C]
    tensor = np.transpose(tensor, (0, 1, 3, 4, 2))

    # 设置绘图
    fig, axes = plt.subplots(B, T, figsize=(12, 3 * B))
    if B == 1:
        axes = np.expand_dims(axes, 0)  # 确保axes是二维的
    if T == 1:
        axes = np.expand_dims(axes, 1)  # 确保axes是二维的

    # 绘制每个批次的每个时间步的图像
    for b in range(B):
        for t in range(T):
            img = tensor[b, t]
            ax = axes[b, t]
            ax.imshow(img)
            ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.show()

class MMNISTDataset(Dataset):
    def __init__(self, data_type):
        # 读取数据
        data = np.load('dataset/mnist_test_seq.npy')

        # 检查数据类型并划分数据集
        if data_type == 'train':
            # 取前9000个样本作为训练集
            self.imgs = data[:10, :9000, :, :]
            self.labels = data[10:, :9000, :, :]
        elif data_type == 'val':
            # 取后1000个样本作为验证集
            self.imgs = data[:10, 9000:, :, :]
            self.labels = data[10:, 9000:, :, :]
        else:
            raise ValueError("data_type must be 'train' or 'val'")

    def __len__(self):
        return self.imgs.shape[1]

    def __getitem__(self, idx):
        # 获取整个时间序列的一部分
        img = self.imgs[:, idx, :, :]
        label = self.labels[:, idx, :, :]

        # 转换为PyTorch张量
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()

        # 归一化图像数据到0-1范围
        img = img / 255.0
        label = label / 255.0

        return img.unsqueeze(1), label.unsqueeze(1)

def calculate_mean_ssim(outputs, labels):
    N, T, C, H, W = outputs.shape
    ssim_sum = 0.0

    # 遍历每个样本和每个时间步
    for n in range(N):
        for t in range(T):
            # 对于单通道图像，确保去除通道维度
            if C == 1:
                output_frame = outputs[n, t, 0, :, :].cpu().numpy()
                label_frame = labels[n, t, 0, :, :].cpu().numpy()
            else:
                # 对于多通道图像，保持通道维度
                output_frame = outputs[n, t, :, :, :].cpu().numpy()
                label_frame = labels[n, t, :, :, :].cpu().numpy()

            # 计算两个帧之间的 SSIM
            ssim_value = cal_ssim(output_frame, label_frame, data_range=label_frame.max() - label_frame.min(), multichannel=C > 1)
            ssim_sum += ssim_value

    # 计算平均 SSIM
    mean_ssim = ssim_sum / (N * T)
    return mean_ssim


def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'dataset/train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist

def load_fixed_set(root):
    # Load the fixed dataset
    filename = 'dataset/mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset

def transform(inputs, label):
    inputs = (inputs > 0.5).float()
    label = (label > 0.5).float()

    inputs = inputs.long()
    inputs = torch.nn.functional.one_hot(inputs, num_classes=2)
    inputs = inputs.squeeze(2)  # 移除原始的通道维度
    inputs = inputs.permute(0, 1,4, 2, 3)

    label_squeezed = label.squeeze(2)

    return inputs, label_squeezed

def transform_MSE(inputs, label):
    inputs = (inputs > 0.5).float()
    label = (label > 0.5).float()

    inputs = inputs.long()
    inputs = torch.nn.functional.one_hot(inputs, num_classes=2).float()
    inputs = inputs.squeeze(2)  # 移除原始的通道维度
    inputs = inputs.permute(0, 1,4, 2, 3)

    label = label.long()
    label = torch.nn.functional.one_hot(label, num_classes=2).float()
    label = label.squeeze(2)  # 移除原始的通道维度
    label = label.permute(0, 1, 4, 2, 3)
    return inputs, label

class MovingMNIST(data.Dataset):
    def __init__(self, root="", is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
                 transform=None):
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_,
                         self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        r = 1
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(
            0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()

        return input, output

    def __len__(self):
        return self.length


def plot_named_tuples(tuples_list):
    # Generate a color map
    colors = plt.cm.jet(np.linspace(0, 1, len(tuples_list)))

    for idx, (x, y, name) in enumerate(tuples_list):
        plt.scatter(x, y, color=colors[idx], label=name)

    plt.xlabel('Training speed(it/s)')
    plt.ylabel('MSE')
    plt.title('MSE of different configuration')
    plt.legend()
    plt.show()


def plot_bar_chart(models, values, title, xlabel, ylabel, margin=1):
    """
    Plots a bar chart with given models and their corresponding values.
    The y-axis starts from a value based on min value minus the margin.

    :param models: List of model names.
    :param values: List of values corresponding to each model.
    :param title: Title of the chart.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param margin: The margin to be subtracted from the min value for y-axis start.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color='skyblue')

    # Dynamic Y-axis limits based on margin
    ymin, ymax = min(values), max(values)
    plt.ylim(max(ymin - margin, 0), ymax + margin)  # Ensure ymin doesn't go below 0

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def MSE(pred, true):
    return (pred - true).pow(2).mean(dim=(0, 1)).sum()

