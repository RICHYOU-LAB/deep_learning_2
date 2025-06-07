import matplotlib as mpl 
mpl.use('Agg')  
import matplotlib.pyplot  as plt 
from torch import nn 
import numpy as np 
import torch 
import os 
import random 
from tqdm import tqdm 
from IPython import display 
 
from models.vgg  import VGG_A 
from models.vgg  import VGG_A_BatchNorm 
from data.loaders  import get_cifar_loader 
 
# ## Constants (parameters) initialization 
device_ids = [0, 1, 2, 3] 
num_workers = 4 
batch_size = 128 
 
# add our package dir to path 
module_path = os.path.dirname(os.getcwd())  
home_path = module_path 
figures_path = os.path.join(home_path,  'reports', 'figures') 
models_path = os.path.join(home_path,  'reports', 'models') 
 
# Create directories if they don't exist 
os.makedirs(figures_path,  exist_ok=True) 
os.makedirs(models_path,  exist_ok=True) 
 
# Make sure you are using the right device. 
os.environ["CUDA_DEVICE_ORDER"]  = "PCI_BUS_ID" 
device = torch.device(f"cuda:{device_ids[0]}"  if torch.cuda.is_available()  else "cpu") 
print(f"Device: {device}") 
if torch.cuda.is_available():  
    print(f"Device name: {torch.cuda.get_device_name(device)}")  
 
# Initialize your data loader and 
# make sure that dataloader works 
# as expected by observing one 
# sample from it. 
train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=num_workers) 
val_loader = get_cifar_loader(train=False, batch_size=batch_size, num_workers=num_workers) 
 
for X, y in train_loader: 
    print(f"Image batch shape: {X.shape}")  
    print(f"Label batch shape: {y.shape}")  
    print(f"Labels: {y[:10]}") 
    break 
 
# This function is used to calculate the accuracy of model classification 
def get_accuracy(model, data_loader): 
    model.eval()  
    correct = 0 
    total = 0 
    with torch.no_grad():  
        for x, y in data_loader: 
            x = x.to(device)  
            y = y.to(device)  
            outputs = model(x) 
            _, predicted = torch.max(outputs.data,  1) 
            total += y.size(0)  
            correct += (predicted == y).sum().item() 
    return correct / total 
 
# Set a random seed to ensure reproducible results 
def set_random_seeds(seed_value=0, device='cpu'): 
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    random.seed(seed_value)  
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)  
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic  = True 
        torch.backends.cudnn.benchmark  = False 
 
# We use this function to complete the entire 
# training process. In order to plot the loss landscape, 
# you need to record the loss value of each step. 
# Of course, as before, you can test your model 
# after drawing a training round and save the curve 
# to observe the training 
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None): 
    model.to(device)  
    learning_curve = [np.nan] * epochs_n 
    train_accuracy_curve = [np.nan] * epochs_n 
    val_accuracy_curve = [np.nan] * epochs_n 
    max_val_accuracy = 0 
    max_val_accuracy_epoch = 0 
 
    batches_n = len(train_loader) 
    losses_list = [] 
    grads = [] 
 
    for epoch in tqdm(range(epochs_n), unit='epoch'): 
        if scheduler is not None: 
            scheduler.step()  
        model.train()  
 
        loss_list = []  # use this to record the loss value of each step 
        grad = []  # use this to record the loss gradient of each step 
        learning_curve[epoch] = 0  # maintain this to plot the training curve 
 
        for data in train_loader: 
            x, y = data 
            x = x.to(device)  
            y = y.to(device)  
            optimizer.zero_grad()  
            prediction = model(x) 
            loss = criterion(prediction, y) 
            # You may need to record some variable values here 
            # if you want to get loss gradient, use 
            # grad = model.classifier[4].weight.grad.clone()  
            loss_list.append(loss.item())  
            loss.backward()  
            if model.classifier[4].weight.grad  is not None: 
                grad.append(model.classifier[4].weight.grad.clone().cpu().numpy())  
            optimizer.step()  
 
            learning_curve[epoch] += loss.item()  
 
        losses_list.append(loss_list)  
        grads.append(grad)  
        display.clear_output(wait=True)  
        f, axes = plt.subplots(1,  2, figsize=(15, 3)) 
 
        learning_curve[epoch] /= batches_n 
        axes[0].plot(learning_curve) 
 
        model.eval()  
        train_acc = get_accuracy(model, train_loader) 
        val_acc = get_accuracy(model, val_loader) 
        train_accuracy_curve[epoch] = train_acc 
        val_accuracy_curve[epoch] = val_acc 
        axes[1].plot(val_accuracy_curve, label='Val Acc') 
        axes[1].plot(train_accuracy_curve, label='Train Acc') 
        axes[1].legend() 
        plt.savefig(os.path.join(figures_path,  f'train_epoch_{epoch}.png')) 
        plt.close()  
 
        if val_acc > max_val_accuracy: 
            max_val_accuracy = val_acc 
            max_val_accuracy_epoch = epoch 
            if best_model_path is not None: 
                torch.save(model.state_dict(),  best_model_path) 
 
    return losses_list, grads 
 
# Train your model 
# feel free to modify 
epo = 20 
loss_save_path = figures_path 
grad_save_path = figures_path 
 
set_random_seeds(seed_value=2020, device=device) 
model = VGG_A() 
lr = 0.001 
optimizer = torch.optim.Adam(model.parameters(),  lr=lr) 
criterion = nn.CrossEntropyLoss() 
best_model_path = os.path.join(models_path,  'best_model.pth')  
losses_list, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=best_model_path) 
 
# Save loss and grads 
for i, loss in enumerate(losses_list): 
    np.savetxt(os.path.join(loss_save_path,  f'loss_epoch_{i}.txt'), loss, fmt='%s', delimiter=' ') 
for i, grad in enumerate(grads): 
    np.savetxt(os.path.join(grad_save_path,  f'grads_epoch_{i}.txt'), np.array(grad).flatten(),  fmt='%s', delimiter=' ') 
 
# Maintain two lists: max_curve and min_curve, 
# select the maximum value of loss in all models 
# on the same step, add it to max_curve, and 
# the minimum value to min_curve 
min_curve = [] 
max_curve = [] 
# Extract min/max across all epochs (by step) 
all_losses = np.concatenate(losses_list)  
max_curve = np.max(all_losses,  axis=0) 
min_curve = np.min(all_losses,  axis=0) 
 
# Use this function to plot the final loss landscape, 
# fill the area between the two curves can use plt.fill_between()  
def plot_loss_landscape(): 
    x = np.arange(len(max_curve))  
    plt.figure(figsize=(10,  5)) 
    plt.fill_between(x,  min_curve, max_curve, color='lightblue', label='Loss Range') 
    plt.plot(max_curve,  label='Max Loss', color='red') 
    plt.plot(min_curve,  label='Min Loss', color='green') 
    plt.title("Loss  Landscape Across Training Steps") 
    plt.xlabel("Training  Step") 
    plt.ylabel("Loss")  
    plt.legend()  
    plt.grid(True)  
    plt.savefig(os.path.join(figures_path,  'loss_landscape.png'))  
    plt.close()  
 
plot_loss_landscape() 
