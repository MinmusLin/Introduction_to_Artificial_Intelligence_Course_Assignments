import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random


def visualize_MNIST_dataset(train_loader):
    # 从训练集中随机选取100张图片和对应的标签
    selected_images = []
    selected_labels = []
    for images, labels in train_loader:
        for image, label in zip(images, labels):
            selected_images.append(image)
            selected_labels.append(label.item())
        if len(selected_images) >= 100:
            break

    # 将选取的图片转换为numpy数组
    selected_images = torch.stack(selected_images).numpy()

    # 创建10行10列的子图
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))

    # 填充子图
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(selected_images[i * 10 + j][0], cmap='gray')
            axes[i, j].set_title(str(selected_labels[i * 10 + j]), fontsize=12, pad=4)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_prediction_results(model, test_loader):
    # 从测试集中随机选取100张图片和对应的标签
    selected_images = []
    selected_labels = []
    predicted_labels = []
    test_loader_list = list(test_loader)

    selected_data = random.sample(test_loader_list, 100)
    with torch.no_grad():
        for data in selected_data:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for image, label, pred in zip(images, labels, predicted):
                selected_images.append(image)
                selected_labels.append(label.item())
                predicted_labels.append(pred.item())

    # 将选取的图片转换为numpy数组
    selected_images = torch.stack(selected_images).numpy()

    # 创建10行10列的子图
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))

    # 填充子图
    for i in range(10):
        for j in range(10):
            index = i * 10 + j
            axes[i, j].imshow(selected_images[index][0], cmap='gray')
            axes[i, j].set_title(f'{selected_labels[index]}, {predicted_labels[index]}', fontsize=12, pad=4)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_incorrect_predictions(model, test_loader):
    # 存储预测错误的样本
    incorrect_samples = []

    # 遍历测试集
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for image, label, pred in zip(images, labels, predicted):
                if len(incorrect_samples) >= 10:
                    break
                if label != pred:
                    incorrect_samples.append((image, label.item(), pred.item()))

    # 创建子图显示预测错误的样本
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    # 填充子图
    for i, (image, label, pred) in enumerate(incorrect_samples):
        ax = axes[i // 5, i % 5]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f'Label: {label}, Prediction: {pred}', fontsize=10, pad=4)
        ax.axis('off')

    plt.tight_layout()
    plt.show()