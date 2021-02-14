from cycleGAN.generator     import Generator
from cycleGAN.discriminator import Discriminator
from cycleGAN.utils import dual_plot

import torch
import torch.nn as nn


real_label = 1.
fake_label = 0.

class GAN():
    def __init__(self, device, lr):
        self.device = device

        self.G = Generator().to(self.device).apply(self.weights_init)
        self.D = Discriminator().to(self.device).apply(self.weights_init)
        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    def weights_init(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    def train_step(self, input_images, target_images, criterion, batch_size, alpha=0.9):
        self.train()

        # Train with all-real batch
        self.D.zero_grad()
        output = self.D(target_images).view(batch_size, -1)
        labels = torch.ones_like(output, dtype=torch.float32, device=self.device)
        D_real_error = criterion(output, labels) * alpha
        D_real_error.backward()

        # Train with all-fake batch
        fake = self.G(input_images)
        output = self.D(fake.detach()).view(batch_size, -1)
        labels.fill_(fake_label)
        D_fake_error = criterion(output, labels) * alpha
        D_fake_error.backward()
        average_D_fake_output = output.mean().item()
        self.D_opt.step()

        # Update Generator
        self.G.zero_grad()
        output = self.D(fake).view(batch_size, -1)
        labels.fill_(real_label)  # fake labels are real for generator cost
        G_error = criterion(output, labels)
        G_error.backward()
        self.G_opt.step()
        
        return fake, average_D_fake_output
    
    def G_zero_grad(self):
        self.G.zero_grad()

    def G_opt_step(self):
        self.G_opt.step()

    def save_weights(self, filename):
        torch.save(self.G.state_dict(), 'weights/' + filename)
        torch.save(self.D.state_dict(), 'weights/D/' + filename)

    def load_weights(self, filename):
        self.G.load_state_dict(torch.load('weights/' + filename))
        self.D.load_state_dict(torch.load('weights/D/' + filename))

    def set_lr(self, lr):
        self.G_opt =  torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.D_opt =  torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

    def train(self):
        self.D = self.D.train()
        self.G = self.G.train()

    def eval(self):
        self.D = self.D.eval()
        self.G = self.G.eval()

class cycleGAN():
    def __init__(self, X_dataloader, Y_dataloader, mode, lr=0.0002):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.mode = mode
        self.lr = lr

        self.X_dataloader = X_dataloader
        self.Y_dataloader = Y_dataloader
        self.X_GAN = GAN(self.device, lr)
        self.Y_GAN = GAN(self.device, lr)

    def train_cycle(self, X_images, Y_images, lmbda=10):
        l1_criterion = nn.L1Loss()
        self.X_GAN.G_zero_grad()
        self.Y_GAN.G_zero_grad()
        loss = (l1_criterion(self.Y_GAN.G(self.X_GAN.G(X_images)), X_images) + l1_criterion(self.X_GAN.G(self.Y_GAN.G(Y_images)), Y_images)) * lmbda
        loss.backward()
        self.X_GAN.G_opt_step()
        self.Y_GAN.G_opt_step()

    def train(self, criterion, batch_size, num_epochs=100, decaying_lr=False):
        print("Starting Training Loop...")
        self.save_weights(self.mode)
        for epoch in range(num_epochs):
            for iters, (X, Y) in enumerate(zip(self.X_dataloader, self.Y_dataloader), 0):
                X, Y = X[0].to(self.device), Y[0].to(self.device)
                # the try-except construction is there not to deal with different sizes of the last batch 
                try:
                    G_X_generated, D_X_output = self.X_GAN.train_step(X.to(self.device), Y.to(self.device), criterion, batch_size)  
                    G_Y_generated, D_Y_output = self.Y_GAN.train_step(Y.to(self.device), X.to(self.device), criterion, batch_size)
                    self.train_cycle(X, Y)
                except:
                    continue

                if (iters % 10 == 0) and (iters % 20 != 0):
                    self.save_weights(self.mode)
                    print("num iters: {}, D_X(Y_fake) average output: {}, D_Y(X_fake) average output:{}, X_real-G_X_generated images:".format(iters, round(D_X_output, 3), round(D_Y_output, 3)))
                    dual_plot(X, G_X_generated)

                if iters % 20 == 0:
                    print("num iters: {}, D_X(Y_fake) average output: {}, D_Y(X_fake) average output:{}, Y_real-G_Y_generated images:".format(iters, round(D_X_output, 3), round(D_Y_output, 3)))
                    dual_plot(Y, G_Y_generated)

            self.update_lr(self.lr, epoch)

        print("Learning finished")

    def save_weights(self, filename):
        filename = filename.replace(' ', '_')
        self.X_GAN.save_weights(filename)
        self.Y_GAN.save_weights('_'.join(list(reversed(filename.split('_')))))

    def load_weights(self, filename):
        filename = filename.replace(' ', '_')
        self.X_GAN.load_weights(filename)
        self.Y_GAN.load_weights('_'.join(list(reversed(filename.split('_')))))

    def update_lr(self, lr, epoch):
        if (epoch > 1) and (epoch % 10 == 0):
            lr /= 1.5
            self.X_GAN.set_lr(lr)
            self.Y_GAN.set_lr(lr)
    
    def transform(self, input, mode='X_to_Y'):
        self.X_GAN.eval()
        self.Y_GAN.eval()
        if mode == 'X_to_Y':
            return self.X_GAN.G(input.to(self.device))
        else:
            return self.Y_GAN.G(input.to(self.device))