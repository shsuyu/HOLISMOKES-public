# This code loads the trained network presented in Schuldt et al. (2023), HOLISMOKES IX, and predicts with it the lens mass parameter values and uncertainties of the provided images of the lens systems.
# Jan 31, 2023 -- Dr. Stefan Schuldt
# Schuldt et al. (2022), arXiv e-prints, arXiv:2206.11279


import numpy as np
import random
import argparse #part of the basic installation within the python distribution
from astropy.io import fits
import torch
import torch.nn as nn


# Residual block
class ResidualBlock(nn.Module):
    
    # 3x3 convolution
    def conv3x3(self, in_channels, out_channels, stride=1):
        #return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
        #                 stride=stride, padding=1, bias=False)
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride, padding=1)
    
    # 1x1 convolution
    def conv1x1(self, in_channels, out_channels, stride=1):
        #this will only scale to more feature maps.
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride, padding=1)
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = self.conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    
# ResNet # number 101
class ResNet101(nn.Module):

    
    # 3x3 convolution
    def conv3x3(self, in_channels, out_channels, stride=1):
        #return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
        #                 stride=stride, padding=1, bias=False)
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride, padding=1)
    
    # 1x1 convolution
    def conv1x1(self, in_channels, out_channels, stride=1):
        #this will only scale to more feature maps.
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride, padding=1)
    def __init__(self, block, layers, filter_amount, num_classes, norm, p, neurons, subfactor, FWHM, stride):
        super(ResNet101, self).__init__()
        self.in_channels = neurons[0]
        self.norm = norm
        self.p = p
        self.npars = num_classes
        self.FWHM = FWHM

        self.conv = self.conv3x3(filter_amount, neurons[0])
        self.bn = nn.BatchNorm2d(neurons[0])
        self.relu = nn.ReLU(inplace=True) #16 channels
        self.layer1 = self.make_layer(block, neurons[1], layers[0], stride[0]) #32 channels
        self.layer2 = self.make_layer(block, neurons[2], layers[1], stride[1])
        self.layer3 = self.make_layer(block, neurons[3], layers[2], stride[2])
        self.avg_pool = nn.AvgPool2d(8)
        if FWHM:
            self.fc = nn.Linear(neurons[3]*(int(8/np.product(stride)*subfactor )**2)+filter_amount, 2*num_classes)
        else:
            self.fc = nn.Linear(neurons[3]*(int(8/np.product(stride)*subfactor )**2), 2*num_classes)
        if self.norm == 'no':
            pass
        elif self.norm == 'softmax':
            self.softmax = nn.Softmax(dim=1)
        elif self.norm == 'sigmoid':
            self.sigmoid = nn.Sigmoid()
        else:
            print('ERROR: -norm command not accepted!\nexiting...')
            sys.exit()
            
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                self.conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.FWHM:
            y = x
            FWHM = y[int(y.shape[0]/2):,:,0,0]
            img = y[:int(y.shape[0]/2),:,:,:]
        else:
            img = x
            
        out = self.conv(img)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        if self.FWHM:
            out = torch.cat((out,FWHM),1)
            
        if self.p !=0:
            out = F.dropout(out, p=self.p, training=True)
        
        out = self.fc(out)
        if self.norm == 'softmax':
            out = self.softmax(out)
        elif self.norm == 'sigmoid':
            out = self.sigmoid(out)        

        mean = out[:, :self.npars]#[:, None] # use first index as the mean
        std = torch.clamp(out[:, self.npars:], min=0.0001) # use second index as the std

        norm_dist = torch.distributions.Normal(mean, std)

        return norm_dist

        #return out


class LensModelNet(object):
    
    def set_parameter(self):

        parser = argparse.ArgumentParser(description="Network for predicting lens mass parameter (Schuldt et al. 2022)")
        parser.add_argument('-cat', type=str, default=False,
                            help="path of catalog with mock IDs")
        parser.add_argument('-o', type=str, default=None,
                            help='path of output catalog')
        parser.add_argument('-col', default=1, type=int,
                            help="Column of IDs in provided catalog (1-based)")
        parser.add_argument('-print', action="store_true",
                            help="Print out predictions for each lens separately")
        parser.add_argument('-data', type=str, default=False,
                            help="path to images")
        args = parser.parse_args()

        if args.cat is False:
            raise NameError('Catalog name required!')
        self.cat = args.cat
        
        if args.o is None:
            self.out = 'pred'
        else:
            self.out = args.o
            
        self.col = args.col
        
        if args.print:
            self.print = True
        else:
            self.print = False

        if args.data is False:
            raise NameError('Path to images required!')
        else:
            self.data = args.data

    def read_cat(self):

        f = open(self.cat, 'r')
        data = f.read().splitlines()
        f.close()

        IDs = []
        for line in data:
            IDs.append(line.split()[self.col-1]) #-1 to make it 1-based
        
        return IDs
    
    def load_image(self, filter_amount, ID):
 
        f = self.data+'/lens_'+str(ID)+'.fits'
        
        hdul = fits.open(f)
        imsize = hdul[1].data.shape
        array = np.zeros((filter_amount, imsize[0], imsize[1]))

        for n in range(filter_amount):
            image = hdul[2*n+1].data
            array[n] = image

        hdul.close()

        return array
    
    def scale_back(self, eta_med, c, npars, eta_std=False):

        # min = a*0 + b
        # max = a*1 + b

        # a = max - min
        # b = min

        if c:
            if npars==7:
                ranges = [(-0.6,0.6), (-0.6,0.6), (-1,1), (-1,1), (0.5,5), (-0.1, 0.1), (-0.1, 0.1)]
            elif npars ==2:
                ranges = [ (-0.1, 0.1), (-0.1, 0.1)]
            else:
                ranges = [(-0.6,0.6), (-0.6,0.6), (-1,1), (-1,1), (0.5,5)]
        else:
            if npars==7:
                ranges = [(-0.6,0.6), (-0.6,0.6), (0,1), (0,np.pi), (0.5,5), (0, 0.2), (0, np.pi)]
            else:
                ranges = [(-0.6,0.6), (-0.6,0.6), (0,1), (0,np.pi), (0.5,5)]
                    
        a = np.zeros(len(ranges))
        b = np.zeros(len(ranges))
        for x in range(len(ranges)):
            a[x] = (ranges[x][1]-ranges[x][0])
            b[x] = ranges[x][0]

        # transform back median
        eta_med = a*eta_med + b
        # since 2021-06-04, also transform uncertainty back
        # do not need to have offset b, since err is wrt to eta_med
        if eta_std is not False:
            eta_std = a*eta_std
        
        return eta_med, eta_std    
            
    def __init__(self):
        self.set_parameter()
        
        _seed = 1
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)

        # specifications of network
        layers = [2,2,2]
        weights = [1,1,1,1,5,1,1]
        norm = 'sigmoid'
        p = 0 # drop out rate
        neuron_list = [16,24,32,64]
        subfactor = 1
        stride = [2,1,1]
        sqrt = False
        FWHM = False
        c = True # complex quantities used
        regconst = 0.5
        BatchSize  = 32
        filter_amount = 4

        net_path = 'final.t7'

        net = ResNet101(ResidualBlock, layers, filter_amount, len(weights), norm, p, neuron_list, subfactor, FWHM, stride)

        # check if the code is running on cpu or gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # force to network to the device the code is running (cpu or gpu)
        net = net.to(device)

        checkpoint = torch.load(net_path, map_location=torch.device(device))
        net.load_state_dict(checkpoint['net'])
        net = net.to(device)
        net.eval()

        #print('Includ the uncertainty scaling')
        scale = (1.25, 1.32, 1.19, 1.20,  1.08, 1.21, 1.21)  # scale predicted uncertainties such that we can interprete them as 1sigma

        #load catalog
        IDs = self.read_cat()

        #record of predictions
        medians = []
        sigmas = []

        print('Predict now SIE+shear parameter values.')
        if self.print:
            print('Print out in the following way:\nlcx, lcy, ex, ey, theta_E, gamma1, gamma2\n')
        
        # iterate over sample
        for ID in IDs:
            if self.print:
                print('start with lens '+str(ID))

            #open image
            image = self.load_image(filter_amount, ID)
            image = torch.from_numpy(image).to(torch.float)
            image = image.to(device)
            image = image[None]

            #apply network to image
            with torch.no_grad():
                output = net(image)

            #split into median and sigma
            median = output.mean
            sigma = output.stddev
    
            median  = median[0].cpu().numpy()
            sigma = sigma[0].cpu().numpy()

            # scale back
            if norm != 'no':
                median, sigma = self.scale_back(median, c, len(weights), sigma)

            #take scaling of errors into account, such that we can interpret them as 1 sigma 
            sigma = sigma*scale

            if self.print:
                print('predicted median values:'+str(median))
                print('predicted uncertainties:' +str(sigma))

            medians.append(median)
            sigmas.append(sigma)

        print('Saving table...')
        np.save(self.out+'_median', medians)
        np.save(self.out+'_sigma', sigmas)
        print('Table saved under:\n\t'+str(self.out)+'_median.npy\n\t'+str(self.out)+'_sigma.npy')

#write README
# add software citations used in paper
            
if __name__ == "__main__":
    LensModelNet()
