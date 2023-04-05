from SMLM.torch import ConvVAE, loss
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torchvision


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(64), transforms.CenterCrop(64), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST("data", download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


def show(x):
    img = x.data.cpu().permute(1, 2, 0).numpy() * 255
    plt.imshow(img)
    plt.show()

#training loop
beta = 2
vae = ConvVAE(3)
opt = torch.optim.Adam(vae.parameters(), lr=5e-4)

for epoch in range(100):
    print(f'Training epoch {epoch}...')
    for i, x in enumerate(loader):
        if len(x) == 2:
            x = x[0]
        #x = x.cuda()
        mu, logvar, out = vae(x)
        rl, kl, l = loss(x, out, mu, logvar, beta)
        opt.zero_grad()
        l.backward()
        opt.step()

        if i == 0:
            vae.eval()
            data = vae.generate(8)
            grid_img = torchvision.utils.make_grid(data, nrow=8, normalize=True)
            show(grid_img)
            vae.train()
