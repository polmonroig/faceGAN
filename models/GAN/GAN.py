"""
GAN.py: Deep convolutional generative adversial network
"""
__author__ = "Pol Monroig Company"

# load libraries
import torch
import torch.nn as nn
import torch.optim as optim
from os.path import join
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, n_channels, imsize):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # in_filters, out_filters, kernel, stride, padding
            nn.Conv2d(n_channels,  imsize, 4, 2, 1), # conv1 64x64
            nn.BatchNorm2d(imsize),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imsize, imsize * 2, kernel_size=4, # conv2 I=32x32
                      stride=2, padding=1),
            nn.BatchNorm2d(imsize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imsize * 2, imsize * 4, kernel_size=4, # conv3 I=16x16
                      stride=2, padding=1),
            nn.BatchNorm2d(imsize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imsize * 4, imsize * 4, kernel_size=4, # conv4 O=2x2
                      stride=2, padding=1),
            nn.BatchNorm2d(imsize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imsize * 4, imsize * 8, kernel_size=4, # conv5
                      stride=2, padding=1),
            nn.BatchNorm2d(imsize * 8),
            nn.LeakyReLU(0.2, inplace=True), # conv6
            nn.Conv2d(imsize * 8, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net(input)





class Generator(nn.Module):
    def __init__(self, latent_size, n_channels, imsize):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, imsize * 8, kernel_size=4, #conv1
                               stride=1, padding=1),
            nn.BatchNorm2d(imsize * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(imsize * 8, imsize * 4, kernel_size=4, #conv2
                               stride=2, padding=1),
            nn.BatchNorm2d(imsize * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(imsize * 4, imsize * 4, kernel_size=4, #conv3
                               stride=2, padding=1),
            nn.BatchNorm2d(imsize * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(imsize * 4, imsize * 2, kernel_size=4, #conv4
                               stride=2, padding=1),
            nn.BatchNorm2d(imsize * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(imsize * 2, imsize,kernel_size=4,#conv5
                               stride=2,padding=1),
            nn.BatchNorm2d(imsize),
            nn.ReLU(True),
            nn.ConvTranspose2d(imsize, n_channels, kernel_size=4,#conv6
                               stride=2, padding=1),
            nn.Tanh()

            )
    def forward(self, input):
        return self.net(input)


class GAN():
    def __init__(self, device, save_dir):
        self.generator = None
        self.discriminator = None
        self.save_dir = save_dir
        self.device = device

    def compile(self, latent_size=100, n_channels=3, imsize=64):

        self.latent_size = latent_size
        self.generator = Generator(latent_size, n_channels, imsize).to(self.device)
        self.discriminator = Discriminator(n_channels, imsize).to(self.device)
        dis_learning_rate = 0.0002
        gen_learning_rate = 0.0002
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(),
                           dis_learning_rate, betas=(0.5, 0.999))
        self.gen_optimizer = optim.Adam(self.generator.parameters(),
                           dis_learning_rate, betas=(0.5, 0.999))

    def load(self, save_dir):
        self.generator.load_state_dict(torch.load(
                join(save_dir, "generator_state.pt")))
        self.discriminator.load_state_dict(torch.load(
                join(save_dir, "discriminator_state.pt")))
        self.discriminator.eval()
        self.generator.eval()

    def  __call__(self, input):
        self.discriminator.eval()
        self.generator.eval()
        return self.generator(input)

    def train(self, dataloader, n_epochs):

        if self.generator is None:
            raise Exception("Compile the model before training")

        criterion = nn.BCELoss()

        # keep track of progress
        self.G_losses = []
        self.D_losses = []

        for epoch in range(n_epochs):

            for i, data in enumerate(dataloader, 0):
                # load image batch
                image, _ = data
                image = image.to(self.device)
                batch_size = image.size()[0]
                trues = torch.ones([batch_size, 1], device=self.device)
                fakes = torch.zeros([batch_size, 1], device=self.device)




                # TRAIN DISCRIMINATIVE
                # zero grads
                self.dis_optimizer.zero_grad()
                # true batch
                out = self.discriminator(image).view(-1, 1)
                loss_real = criterion(out, trues)
                loss_real.backward(retain_graph=True)
                accuracy_dis = out.mean().item()

                # fake batch
                noise = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)

                fake = self.generator(noise)
                out = self.discriminator(fake).view(-1, 1)
                loss_fake = criterion(out, fakes)
                loss_fake.backward(retain_graph=True)
                lossD = loss_real + loss_fake
                self.dis_optimizer.step()
                accuracy_gen_1 = out.mean().item()

                # TRAIN GENERATOR
                # zero grads
                self.gen_optimizer.zero_grad()

                out = self.discriminator(fake).view(-1, 1)
                lossG = criterion(out, trues)
                lossG.backward()
                self.gen_optimizer.step()
                accuracy_gen_2 = out.mean().item()

                # PRINT STATS
                if i % 100 == 0:
                    torch.save(self.discriminator.state_dict(),
                               join(self.save_dir, "discriminator_state.pt"))
                    torch.save(self.generator.state_dict(),
                               join(self.save_dir, "generator_state.pt"))
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tAcc_dis: %.4f\tAcc_gen: %.4f / %.4f'
                  % (epoch, n_epochs, i, len(dataloader),
                     lossD.item(), lossG.item(), accuracy_dis, accuracy_gen_1, accuracy_gen_2))

                self.G_losses.append(lossG.item())
                self.D_losses.append(lossD.item())

    def plot(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
