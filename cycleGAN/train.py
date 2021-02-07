from cycleGAN.generator     import Generator
from cycleGAN.discriminator import Discriminator
from cycleGAN.utils         import weights_init
import torch
import torch.nn as nn


real_label = 1.
fake_label = 0.

def get_GAN(lr, beta, device):
    G = Generator().to(device).apply(weights_init)
    D = Discriminator().to(device).apply(weights_init)

    G_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta, 0.999))
    D_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta, 0.999))

    return (G, D), (G_opt, D_opt)

def train_GAN(D, G, D_opt, G_opt, starting_images, target_images, batch_size, criterion, device, alpha=1.1):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    D.zero_grad()
    output = D(target_images).view(batch_size, -1)
    labels = torch.ones_like(output, dtype=torch.float32, device=device)
    D_real_error = criterion(output, labels) * alpha
    D_real_error.backward()

    ## Train with all-fake batch
    fake = G(starting_images)
    output = D(fake.detach()).view(batch_size, -1)
    labels.fill_(fake_label)
    D_fake_error = criterion(output, labels) * alpha
    D_fake_error.backward()
    average_D_fake_output = output.mean().item()
    D_opt.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    G.zero_grad()
    output = D(fake).view(batch_size, -1)
    labels.fill_(real_label)  # fake labels are real for generator cost
    G_error = criterion(output, labels)
    G_error.backward()
    G_opt.step()
    
    return fake.to('cpu').detach()[0, :, :, :].permute(1,2,0).numpy(), (G_error.item(), average_D_fake_output)

def train_cycle(G_X, G_Y, G_X_opt, G_Y_opt, X_images, Y_images, lmbda=25):
    l1_criterion = nn.L1Loss()
    G_X.zero_grad()
    G_Y.zero_grad()
    loss = (l1_criterion(G_Y(G_X(X_images)), X_images) + l1_criterion(G_X(G_Y(Y_images)), Y_images)) * lmbda
    loss.backward()
    G_X_opt.step()
    G_Y_opt.step()