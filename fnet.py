import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.fft import fft
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class FBlock(nn.Module):
    def __init__(self, dims, p):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.feedforward = nn.Sequential(
            nn.Linear(dims, dims * 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(dims * 2, dims),
        )

    def forward(self, x):
        x = self.norm1((fft(fft(x, dim=-1), dim=-2)).real + x)
        x = self.norm2(self.feedforward(x) + x)
        return x


class FNet(nn.Module):
    def __init__(self, dims, p=0.1, blocks=3, classes=10):
        super(FNet, self).__init__()
        lay = [FBlock(dims, p) for _ in range(blocks)]
        self.block = nn.Sequential(*lay)
        self.clf = nn.Sequential(
            nn.Linear(dims, dims * 2),
            nn.GELU(),
            nn.Linear(dims * 2, classes)
        )

    def forward(self, x):
        x = self.block(x).mean(dim=1)
        return self.clf(x)

class VisionFNet(nn.Module):
    def __init__(self, patch_size=7, channel=1, dims=64, classes=10, p=0.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(channel, dims, patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        self.fnet = FNet(dims=dims, p=p, blocks=3, classes=classes)

    def forward(self, x):
        return self.fnet(self.embed(x))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = MNIST('data/', train=True, download=True, transform=transform)
    test_ds = MNIST('data/', train=False, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)

    model = VisionFNet(patch_size=7, channel=1, dims=64, classes=10)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        loss_avg = 0.
        for idx, (img, tar) in enumerate(train_dl, start=1):
            out = model(img)
            loss = criterion(input=out, target=tar)
            loss_avg += (loss.item() - loss_avg) / idx

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        correct = 0
        for idx, (img, tar) in enumerate(test_dl):
            with torch.no_grad():
                out = model(img)
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(tar.view_as(pred)).sum().item()

        s_epoch = f"[{epoch:2d} / {num_epochs}] "
        s_loss = f"tain {loss_avg=:.12f} "
        s_acc = f"test acc: {correct / len(test_dl.dataset)}"
        print(s_epoch + s_loss + s_acc)
