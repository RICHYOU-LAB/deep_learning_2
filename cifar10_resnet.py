import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 1. 数据加载与增强
# ---------------------------
def get_data_loaders():
    """创建带数据增强的数据加载器"""
    # 训练集增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # 测试集转换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # 下载/加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader

# ---------------------------
# 2. 网络模型定义 (满足要求2a-d, 3a-c)
# ---------------------------
class ResidualBlock(nn.Module):
    """残差块 (满足要求3c)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化 (满足要求3a)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        return F.relu(out)

class CIFAR10ResNet(nn.Module):
    """CIFAR-10专用ResNet模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块堆叠
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        # 分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)  # Dropout (满足要求3b)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # 应用Dropout
        return self.fc(x)

# ---------------------------
# 3. 训练与评估函数
# ---------------------------
def train_model(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=50):
    """训练模型并记录指标"""
    train_losses, train_accs, test_accs, lr_history = [], [], [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        start_time = time.time()
        
        # 训练循环
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练指标
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 评估测试集
        test_acc = evaluate_model(model, testloader)
        test_accs.append(test_acc)
        
        # 记录学习率
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_acc)
            else:
                scheduler.step()
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | LR: {lr_history[-1]:.6f}")
    
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    return train_losses, train_accs, test_accs, lr_history

def evaluate_model(model, testloader):
    """评估模型性能"""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# ---------------------------
# 4. 可视化函数 (满足要求6)
# ---------------------------
def visualize_filters(layer, title="Conv1 Filters"):
    """可视化卷积层滤波器"""
    filters = layer.weight.data.cpu().numpy()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            # 将滤波器转换为图像格式 (C, H, W) -> (H, W, C)
            f = filters[i].transpose(1, 2, 0)
            # 归一化显示
            f_min, f_max = np.min(f), np.max(f)
            f = (f - f_min) / (f_max - f_min)
            ax.imshow(f)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('conv_filters.png')
    plt.show()

def plot_metrics(train_losses, train_accs, test_accs, lr_history):
    """绘制训练指标曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    # 学习率曲线
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('learning_rate.png')
    plt.show()

def visualize_activations(model, testloader):
    """可视化网络激活图"""
    model.eval()
    data_iter = iter(testloader)
    images, _ = next(data_iter)
    image = images[0].unsqueeze(0).to(device)  # 取第一张图像
    
    # 注册钩子获取激活
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册钩子
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, ResidualBlock):
            hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # 前向传播
    with torch.no_grad():
        model(image)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化激活
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    layer_names = list(activations.keys())[:6]  # 取前6层
    
    for i, name in enumerate(layer_names):
        ax = axes[i//3, i%3]
        act = activations[name].cpu().squeeze()
        
        # 取前8个通道的平均激活
        if len(act.shape) == 4:
            act = act[0]  # 取batch中第一个
        channel_avg = torch.mean(act[:8], dim=0)
        
        ax.imshow(channel_avg, cmap='viridis')
        ax.set_title(f'{name} Activation')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('layer_activations.png')
    plt.show()

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    # 1. 准备数据
    trainloader, testloader = get_data_loaders()
    
    # 2. 初始化模型 (满足要求2a-d)
    model = CIFAR10ResNet().to(device)
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 配置损失函数和优化器 (满足要求4b, 5a)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.SGD(model.parameters(), lr=0.1, 
                         momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                 patience=5)
    
    # 4. 训练模型
    print("Starting training...")
    train_losses, train_accs, test_accs, lr_history = train_model(
        model, trainloader, testloader, criterion, optimizer, scheduler, epochs=50
    )
    
    # 5. 最终评估
    final_acc = evaluate_model(model, testloader)
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    
    # 6. 可视化分析 (满足要求6)
    # 可视化第一层卷积滤波器
    visualize_filters(model.conv1)
    
    # 绘制训练指标
    plot_metrics(train_losses, train_accs, test_accs, lr_history)
    
    # 可视化激活图
    visualize_activations(model, testloader)
    
    # 7. 保存最终模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")

if __name__ == "__main__":
    main()