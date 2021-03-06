import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


os.makedirs("images", exist_ok=True)

'''
-Se salveaza progresul odata la 100 epoci (se poate modifica numarul de epoci)
-Pentru a face load se pune in default la variabilele resumeG si resumeD PATH-ul catre fisierele unde a fost salvat progresul (EX: D:\Ceva\checkpointGenerator.pth)
-Daca resumeG si resumeD sunt empty ( '' ) nu se va incerca load

'''

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lrG", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lrD", type=float, default=0.0002, help="adam: learning rate")
#parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--wd", type=float, default=0.1, help="adam: weight decay")
#parser.add_argument("--epsilon", type=float, default=0.00000001, help="adam: very small number to prevent any division by zero in the implementation")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=200, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument('--resumeG', type=str, default='', metavar='PATH', help='path to latest checkpoint for Generator(default: none)')
parser.add_argument('--resumeD', type=str, default='', metavar='PATH', help='path to latest checkpoint for Generator(default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
    
    
class facadeDataset(Dataset):
    
    #initializare clasa cu numPyArray-ul create la Partea 1
    def __init__(self, finalNumpyArray):
        super(facadeDataset).__init__()
        self.finalNumpyArray = finalNumpyArray
        
    #functie care returneaza numarul de elemente din Dataset        
    def __len__(self):        
        return len(self.finalNumpyArray)
    
    #functie care returneaza un item din Dataset
    def __getitem__(self, index):
        
        image =  self.finalNumpyArray[index]
        result = self.transform(image)
        return result
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(opt.img_size,opt.img_size)),
        #T.RandomResizedCrop(image_size),
        #T.RandomHorizontalFlip(),
        transforms.ToTensor()])


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

#--------------------------------------------------------------De Terminat

def save_checkpoint_generator(state, filename='checkpointGenerator.pth'):
    torch.save(state, filename)
    
def save_checkpoint_discriminator(state, filename='checkpointDiscriminator.pth'):
    torch.save(state, filename)

# Load checkpoint if exists
if opt.resumeG and opt.resumeD:
    if os.path.isfile(opt.resumeG) and os.path.isfile(opt.resumeD):
        
        print("=> loading checkpoint ")
        
        #Load Generator
        checkpointG = torch.load(opt.resumeG)
        opt.start_epoch = checkpointG['epoch']
        generator.load_state_dict(checkpointG['state_dict'])
        
        #Load Discriminator
        checkpointD = torch.load(opt.resumeD)
        discriminator.load_state_dict(checkpointD['state_dict'])
        
        print("=> loaded checkpoint ")
    else:
        print("=> no checkpoint found at ")



if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
facade = facadeDataset(np.load('E:\Andrei\PEX - NTT\Poze\imgDatasetProcessed\imgDatasetProcessed.npy',allow_pickle=True))

dataloader = torch.utils.data.DataLoader(facade,batch_size=opt.batch_size,shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------



for epoch in range(opt.start_epoch, opt.n_epochs):
    for i, sample in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(sample.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(sample.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(sample.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (sample.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()
        optimizer_G.step()
    

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            
    if epoch % 100 == 0 and epoch > 99:
        print("=> Saving checkpoint ")
        save_checkpoint_generator({
            'epoch': epoch,
            'state_dict': generator.state_dict()
            })
        save_checkpoint_discriminator({
            'state_dict': discriminator.state_dict()
            })
