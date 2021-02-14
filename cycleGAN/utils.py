import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pylab import rcParams
from os import walk

import random
def set_seed(manualSeed=42):
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

def get_GAN_dataloader(mode, batch_size, image_size, workers):
    X_folder, Y_folder = mode.split(' to ')
    X_folder, Y_folder = 'data/' + X_folder, 'data/' + Y_folder


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

def plot(image):
    if (type(image) is tuple) or (type(image) is list):
        plt.imshow(image[0][0, :, :, :].permute(1,2,0).detach().to('cpu').numpy(), cmap='Greys')
    else:
        plt.imshow(image.detach().to('cpu')[0].squeeze(0).permute(1,2,0).numpy(), cmap='Greys')

def dual_plot(image1, image2):
    plt.subplot(1,2,1)
    plot(image1)

    plt.subplot(1,2,2)
    plot(image2)

    plt.show()

def plot_best(mode):
    dirpath = 'example/' + mode
    _, _, filenames = next(walk(dirpath))
    images_filenames = [dirpath + '/' + filename for filename in filenames if filename.split('.')[-1] == 'png']

    ax = []

    n = len(images_filenames)
    rcParams['figure.figsize'] = 9, 4 * n

    for i in range(n):
        ax.append(plt.subplot2grid((n, 1), (i,0)))

    for plot, filename in zip(ax, images_filenames):
        img = plt.imread(filename)
        plot.imshow(img)