import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import random
def set_seed():
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

def get_GAN_dataloader(X_folder, Y_folder, batch_size, image_size, workers):
    X_dataset = dset.ImageFolder(root=X_folder,
                            transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor()
                           ]))
    Y_dataset = dset.ImageFolder(root=Y_folder,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor()
                            ]))
    
    X_dataloader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    Y_dataloader = torch.utils.data.DataLoader(Y_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    return X_dataloader, Y_dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_data(G_X, D_X, G_Y, D_Y, filename):
    torch.save(G_X.state_dict(), 'weights/' + 'G_X' + filename)
    torch.save(G_Y.state_dict(), 'weights/' + 'G_Y' + filename)
    torch.save(D_X.state_dict(), 'weights/' + 'D_X' + filename)
    torch.save(D_Y.state_dict(), 'weights/' + 'D_Y' + filename)

def load_data(G_X, D_X, G_Y, D_Y, filename):
    G_X.load_state_dict(torch.load('weights/' + 'G_X' + filename))
    G_Y.load_state_dict(torch.load('weights/' + 'G_Y' + filename))
    D_X.load_state_dict(torch.load('weights/' + 'D_X' + filename))
    D_Y.load_state_dict(torch.load('weights/' + 'D_Y' + filename))

def plot(image):
    if (type(image) is tuple) or (type(image) is list):
        plt.imshow(image[0][0, :, :, :].permute(1,2,0), cmap='Greys')
    else:
        plt.imshow(image, cmap='Greys')

def dual_plot(image1, image2):
    plt.subplot(1,2,1)
    plot(image1)

    plt.subplot(1,2,2)
    plot(image2)

    plt.show()

def update_weights(D_X, G_X, D_Y, G_Y, X_GAN_loss, Y_GAN_loss):
    X_discriminator_loss = X_GAN_loss[-1][1]
    Y_discriminator_loss = Y_GAN_loss[-1][1]

    if X_GAN_loss[-1][1] + Y_GAN_loss[-1][1] < 2:
        save_data(G_X, D_X, G_Y, D_Y, '')
    else:
        load_data(G_X, D_X, G_Y, D_Y, '')

def update_lr(D_X, G_X, D_Y, G_Y, D_X_opt, G_X_opt, D_Y_opt, G_Y_opt, lr, beta, epoch):
    if (epoch > 1) and (epoch % 5 == 0) and (lr >= 0.0002):
        lr /= 1.5
        D_X_opt = torch.optim.Adam(D_X.parameters(), lr=lr, betas=(beta, 0.999))
        G_X_opt = torch.optim.Adam(G_X.parameters(), lr=lr, betas=(beta, 0.999))

        D_Y_opt = torch.optim.Adam(D_Y.parameters(), lr=lr, betas=(beta, 0.999))
        G_Y_opt = torch.optim.Adam(G_Y.parameters(), lr=lr, betas=(beta, 0.999))