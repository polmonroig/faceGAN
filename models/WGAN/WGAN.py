"""
WGAN.py: Deep convolutional
        generative adversial network with Wasserstein loss
"""
__author__ = "Pol Monroig Company"

# load libraries
import torch
import torch.nn as nn
import torch.optim as optim
from os.path import join
import matplotlib.pyplot as plt
import time as t

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
        )

    def forward(self, input):
        return self.net(input)

    def change_grad(self, grad):
        for p in self.parameters():
            p.requires_grad = grad



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


class WGAN():
    def __init__(self, device, save_dir):
        self.generator = None
        self.discriminator = None
        self.save_dir = save_dir
        self.device = device

    def compile(self, latent_size=100, n_channels=3, imsize=64, wc=0.01):

        self.latent_size = latent_size
        self.weight_cliping_limit = wc
        self.generator = Generator(latent_size, n_channels, imsize).to(self.device)
        self.discriminator = Discriminator(n_channels, imsize).to(self.device)
        dis_learning_rate = 0.00005
        gen_learning_rate = 0.00005
        self.dis_optimizer = optim.RMSprop(self.discriminator.parameters(),
                           dis_learning_rate)
        self.gen_optimizer = optim.RMSprop(self.generator.parameters(),
                           dis_learning_rate)

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

    def get_infinite_batches(self, dataloader):
        while True:
            for i, (images, _) in enumerate(dataloader):
                yield images

    def train(self, dataloader, gen_epochs, critic_epochs):

        if self.generator is None:
            raise Exception("Compile the model before training")
        t_begin = t.time()
        # make batches callable
        data = self.get_infinite_batches(dataloader)


        # keep track of progress
        self.G_losses = []
        self.D_losses = []

        for i in range(gen_epochs):
            lossD = 0

            # activate grad for critic
            self.discriminator.change_grad(True)

            # train critic
            for j in range(critic_epochs):
                # load image batch
                image = data.__next__()
                image = image.to(self.device)
                batch_size = image.size()[0]
                lossD, loss_real, loss_fake = self.train_critic(image, batch_size)

            # deactivate grad for critic
            self.discriminator.change_grad(False)

            # train generator
            lossG = self.train_generator(batch_size)



            # PRINT STATS
            if i % 100 == 0:
                # save model
                torch.save(self.discriminator.state_dict(),
                           join(self.save_dir, "discriminator_state.pt"))
                torch.save(self.generator.state_dict(),
                           join(self.save_dir, "generator_state.pt"))

                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_D_fake: %.4f\tLoss_D_real: %.4f\t'
              % (i, gen_epochs, lossD.item(), lossG.item(), loss_fake, loss_real))

            self.G_losses.append(lossG.item())
            self.D_losses.append(lossD.item())

        t_end = t.time()
        print('Time of training{}'.format((t_end - t_begin)))


    def train_critic(self, image, batch_size):
        # in wasserstein true means 1 and false -1 since we do not use the
        # sigmoid activation function anymore
        # trues = torch.ones([batch_size, 1], device=self.device).float()
        # fakes = torch.tensor([batch_size, -1], device=self.device).float()

        # TRAIN DISCRIMINATIVE
        # zero grads
        self.dis_optimizer.zero_grad()
        # true batch
        out = self.discriminator(image).view(-1, 1)
        loss_real = -torch.mean(out)
        loss_real.backward(retain_graph=True)

        # fake batch
        noise = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)

        fake = self.generator(noise)
        out = self.discriminator(fake).view(-1, 1)
        loss_fake = torch.mean(out)
        loss_fake.backward(retain_graph=True)
        lossD = loss_real + loss_fake
        self.dis_optimizer.step()

        # WEIGHT CLIPPING
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        return lossD, loss_real, loss_fake

    def train_generator(self, batch_size):

        # trues = torch.ones([batch_size, 1], device=self.device)

        # generate fake images
        noise = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
        fake = self.generator(noise)

        # TRAIN GENERATOR
        # zero grads
        self.gen_optimizer.zero_grad()

        out = self.discriminator(fake).view(-1, 1)
        lossG = -torch.mean(out)

        lossG.backward()
        self.gen_optimizer.step()


        return lossG


    def plot(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
