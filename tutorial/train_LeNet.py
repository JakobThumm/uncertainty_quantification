# train_lenet_mnist.py
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------- LeNet (same as before) ----------------
class LeNetPT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: [B,1,32,32]
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)   # 32->28
        self.conv2 = nn.Conv2d(6,16, kernel_size=5, stride=1, padding=0)   # 14->10
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

        # Optional init (good defaults, not mandatory)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B,1,32,32]
        x = F.relu(self.conv1(x))                   # [B,6,28,28]
        x = F.avg_pool2d(x, kernel_size=2)         # [B,6,14,14]
        x = F.relu(self.conv2(x))                   # [B,16,10,10]
        x = F.avg_pool2d(x, kernel_size=2)         # [B,16,5,5]
        x = torch.flatten(x, 1)                     # [B,400]
        x = F.relu(self.fc1(x))                     # [B,120]
        x = F.relu(self.fc2(x))                     # [B,84]
        x = self.fc3(x)                             # [B,10]
        return x

# ---------------- Training script ----------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloaders(batch_size=128, num_workers=2):
    # MNIST stats
    mean, std = 0.1307, 0.3081
    train_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    test_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_tf)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    test_ld  = DataLoader(test_ds, batch_size=512, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    return train_ld, test_ld

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def train(epochs=5, lr=1e-3, batch_size=128, ckpt_path="lenet_pt.pth"):
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ld, test_ld = get_dataloaders(batch_size)
    model = LeNetPT(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(train_ld, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch} [{i:4d}/{len(train_ld)}]  "
                      f"loss: {running_loss / i:.4f}")

        val_loss, val_acc = evaluate(model, test_ld, device)
        print(f"Epoch {epoch} | val_loss: {val_loss:.4f}  val_acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✅ Saved best model to {ckpt_path} (acc={best_acc*100:.2f}%)")

    print(f"Training done. Best test acc: {best_acc*100:.2f}%")
    return ckpt_path

if __name__ == "__main__":
    # Tune epochs if you want ~99%+: try 10-14 epochs.
    train(epochs=8, lr=1e-3, batch_size=128, ckpt_path="lenet_pt.pth")
