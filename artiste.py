import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image
import cv2
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

if __name__ == "__main__":
    
    manualSeed = 1453
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataRoute = "data/train"
    workers = 0
    batch_size = 128
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    numEpochs = 1024 + 1
    lr = 0.00002
    beta1 = 0.5
    ngpu = 1

    
    dataset = dset.ImageFolder(root=dataRoute,
                               transform=transforms.Compose([
                                         transforms.Resize(image_size),
                                         transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    

    real_batch = next(iter(dataloader))
    """
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], 
                            padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    """



    G = Generator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        G = nn.DataParallel(G, list(range(ngpu)))

    G.apply(weights_init)

    print(G)

    D = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        D = nn.DataParallel(netD, list(range(ngpu)))

    D.apply(weights_init)

    print(D)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    rLabel = 1
    fLabel = 0

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    data_directory = "results"
    plot_directory = "plots"

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)


    if (torch.cuda.is_available() and (ngpu > 0)):
        print("Cuda is enabled")


    print("Train Start")

    for epoch in range(numEpochs):
        time_epoch_start = time.time()
        
        for i, data in enumerate(dataloader, 0):
            D.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), rLabel, dtype=torch.float, device=device)

            output = D(real_cpu).view(-1)

            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()


            noise = torch.randn(b_size, nz, 1, 1, device=device)

            fake = G(noise)
            label.fill_(fLabel)

            output = D(fake.detach()).view(-1)
        
            errD_fake = criterion(output, label)
        
            errD_fake.backward()
            D_G_z1 = output.mean().item()
        
            errD = errD_real + errD_fake
        
            optimizerD.step()


            G.zero_grad()
            label.fill_(rLabel)
        
            output = D(fake).view(-1)
        
            errG = criterion(output, label)
        
            errG.backward()
            D_G_z2 = output.mean().item()
        
            optimizerG.step()


            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == numEpochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        time_epoch_end = time.time()
        epoch_time = time_epoch_end - time_epoch_start

        msg = f"[{epoch}/{numEpochs}]   Loss_D: {errD.item():.4f}    "
        msg += f"Loss_G: {errG.item():.4f}    D(x): {D_x:.4f}    "
        msg += f"D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}    "
        msg += f"Epoch time: {epoch_time:.3f}"
        print(msg)
 
        if (epoch % 32) == 0:
            temp_data = {}
            temp_data["G_losses"] = G_losses
            temp_data["D_losses"] = D_losses
            temp_data["img_list"] = img_list

            filename = f"{data_directory}/data_{epoch}e.npy"
            
            np.save(filename, temp_data)

            plt.figure(figsize=(10,5))
            plt.title(f"Generator and Discriminator Loss During Training {epoch}")
            plt.plot(temp_data["G_losses"],label="G")
            plt.plot(temp_data["D_losses"],label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()

            plt.savefig(f"{plot_directory}/gen_dis_loss_{epoch}.png")
            plt.close()


            plt.figure(figsize=(15,15))
            plt.axis("off")
            plt.title(f"Fake Images {epoch}")
            plt.imshow(np.transpose(temp_data["img_list"][-1],(1,2,0)))

            plt.savefig(f"{plot_directory}/images_{epoch}.png")
            plt.close()

            temp_data = {}





    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], 
                            padding=5, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

    
    for img in img_list:
        model.device()
        output = model(img)
        output = output.cpu().numpy()
        cv2.imwrite(output, "image.png")
    """

