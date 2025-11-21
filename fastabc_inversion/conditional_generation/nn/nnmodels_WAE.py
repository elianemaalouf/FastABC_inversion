# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class netD(nn.Module):
    def __init__(self, nc = 1, ndf = 64, dfs = 5, ngpu = 1, ndata = 10, npx = 32, npy = 32, dz = 3):
        super(netD, self).__init__()
        
        self.ngpu = ngpu
        self.ndata = ndata
        self.npx = npx
        self.npy = npy 
        self.dz = dz
        self.dfs = dfs
        self.ndf = ndf 
        self.nc = nc
            
        self.img_conv2d_1 = nn.Conv2d(self.nc, self.ndf, self.dfs, 2, self.dfs//2, bias=False)
        self.img_conv2d_bn2d_1 = nn.BatchNorm2d(self.ndf)
            
        self.img_conv2d_2 = nn.Conv2d(self.ndf, self.ndf*2, self.dfs, 2, self.dfs//2, bias=False)
        self.img_conv2d_bn2d_2 = nn.BatchNorm2d(self.ndf*2)
            
        self.img_conv2d_3 = nn.Conv2d(self.ndf*2, self.ndf*4, self.dfs, 2, self.dfs//2, bias=False)
        self.img_conv2d_bn2d_3 = nn.BatchNorm2d(self.ndf*4)
            
        self.img_conv2d_4 = nn.Conv2d(self.ndf * 4, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.img_conv2d_bn2d_4 = nn.BatchNorm2d(1)
            
                    
        self.measure_ln_1 = nn.Linear(self.ndata, self.ndata)
        self.measure_ln_bn_1 = nn.BatchNorm1d(self.ndata)
            
        self.measure_ln_2 = nn.Linear(self.ndata, self.ndata)
        self.measure_ln_bn_2 = nn.BatchNorm1d(self.ndata)
            
        self.measure_ln_3 = nn.Linear(self.ndata, self.ndata)
        self.measure_ln_bn_3 = nn.BatchNorm1d(self.ndata)
            
        self.code_ln_1 = nn.Linear(self.ndata+16, self.ndata*2)
        self.code_ln_bn_1 =  nn.BatchNorm1d(self.ndata*2)
            
        self.code_ln_2 = nn.Linear(self.ndata*2, self.ndata*2)
        self.code_ln_bn_2 =  nn.BatchNorm1d(self.ndata*2)
            
        self.code_ln_3 = nn.Linear(self.ndata*2, self.ndata)
        self.code_ln_bn_3 =  nn.BatchNorm1d(self.ndata)
            
        self.code_ln_4 = nn.Linear(self.ndata, self.dz* 4)
        self.code_ln_bn_4 =  nn.BatchNorm1d(self.dz* 4)
            
        self.code_ln_5 = nn.Linear(self.dz* 4, self.dz* 2)
        self.code_ln_bn_5 =  nn.BatchNorm1d(self.dz* 2)
            
        self.code_ln_6 = nn.Linear(self.dz * 2, self.dz)
    
    # weight_init
    
    def weight_init(self, fun):
        for m in self._modules:
            fun(self._modules[m])
    

    def forward(self, input_images, input_labels):
        # takes inputs:  image (1 x 32 x 32) and onehot encoded class (10)

        imgcode = F.leaky_relu(self.img_conv2d_bn2d_1(self.img_conv2d_1(input_images)), 0.2, inplace = True)
        imgcode = F.leaky_relu(self.img_conv2d_bn2d_2(self.img_conv2d_2(imgcode)), 0.2, inplace = True)
        imgcode = F.leaky_relu(self.img_conv2d_bn2d_3(self.img_conv2d_3(imgcode)), 0.2, inplace = True)
        imgcode = F.leaky_relu(self.img_conv2d_bn2d_4(self.img_conv2d_4(imgcode)), 0.2, inplace = True)
        imgcode = imgcode.view(-1,16)
        
        measurecode = F.leaky_relu(self.measure_ln_bn_1(self.measure_ln_1(input_labels)), 0.2, inplace = True)
        measurecode = F.leaky_relu(self.measure_ln_bn_2(self.measure_ln_2(measurecode)), 0.2, inplace = True)
        measurecode = F.leaky_relu(self.measure_ln_bn_3(self.measure_ln_3(measurecode)), 0.2, inplace = True)
        
        fullcode = torch.cat((imgcode, measurecode),1)
        fullcode = F.leaky_relu(self.code_ln_bn_1(self.code_ln_1(fullcode)), 0.2, inplace = True)
        fullcode = F.leaky_relu(self.code_ln_bn_2(self.code_ln_2(fullcode)), 0.2, inplace = True)
        fullcode = F.leaky_relu(self.code_ln_bn_3(self.code_ln_3(fullcode)), 0.2, inplace = True)
        fullcode = F.leaky_relu(self.code_ln_bn_4(self.code_ln_4(fullcode)), 0.2, inplace = True)
        fullcode = F.leaky_relu(self.code_ln_bn_5(self.code_ln_5(fullcode)), 0.2, inplace = True)
        fullcode = self.code_ln_6(fullcode)
        
        return fullcode
  

    
class netG(nn.Module):
    def __init__(self, nc = 1, ngf = 64, ngpu = 1, ndata=10, dz = 3, npx = 32 , npy = 32):
        super(netG, self).__init__()
        self.ngpu = ngpu
        self.dz = dz
        self.ndata = ndata
        self.nc = nc
        self.ngf = ngf
        self.npx = npx
        self.npy = npy
        
        self.gen_img_ln_1 = nn.Linear(self.dz, self.dz * 2)
        self.gen_img_bn_ln_1 = nn.BatchNorm1d(self.dz * 2)
        
        self.gen_img_ln_2 = nn.Linear(self.dz * 2, self.dz * 4)
        self.gen_img_bn_ln_2 = nn.BatchNorm1d(self.dz * 4)
        
        self.gen_img_ln_3 = nn.Linear(self.dz * 4, self.dz * 8)
        self.gen_img_bn_ln_3 = nn.BatchNorm1d(self.dz * 8)
        
        self.gen_img_ln_4 = nn.Linear(self.dz* 8, self.npx*self.npy)
        self.gen_img_bn_ln_4 = nn.BatchNorm1d(self.npx*self.npy)
        
        self.gen_img_ln_5 = nn.Linear(self.npx*self.npy, self.npx*self.npy)
        
        self.gen_measure_ln_1 = nn.Linear(self.dz, self.dz * 2)
        self.gen_measure_bn_ln_1 = nn.BatchNorm1d(self.dz * 2)
        
        self.gen_measure_ln_2 = nn.Linear(self.dz* 2, self.dz* 4)
        self.gen_measure_bn_ln_2 = nn.BatchNorm1d(self.dz * 4)
        
        self.gen_measure_ln_3 = nn.Linear(self.dz * 4, self.ndata)
        self.gen_measure_bn_ln_3 = nn.BatchNorm1d(self.ndata)
        
        self.gen_measure_ln_4 = nn.Linear(self.ndata, self.ndata)
        self.gen_measure_bn_ln_4 = nn.BatchNorm1d(self.ndata)
        
        self.gen_measure_ln_5 = nn.Linear(self.ndata, self.ndata)
        self.gen_measure_bn_ln_5 = nn.BatchNorm1d(self.ndata)
        
    # weight_init
    
    def weight_init(self, fun):
        for m in self._modules:
            fun(self._modules[m])  


    def forward(self, code):
        
        img = F.relu(self.gen_img_bn_ln_1(self.gen_img_ln_1(code)), True)
        img = F.relu(self.gen_img_bn_ln_2(self.gen_img_ln_2(img)), True)
        img = F.relu(self.gen_img_bn_ln_3(self.gen_img_ln_3(img)), True) 
        img = F.relu(self.gen_img_bn_ln_4(self.gen_img_ln_4(img)), True)
        img = self.gen_img_ln_5(img)
        img = img.view(-1, self.nc, self.npx, self.npy)
        
        measure = F.relu(self.gen_measure_bn_ln_1(self.gen_measure_ln_1(code)), True)
        measure = F.relu(self.gen_measure_bn_ln_2(self.gen_measure_ln_2(measure)), True)
        measure = F.relu(self.gen_measure_bn_ln_3(self.gen_measure_ln_3(measure)), True)
        measure = F.relu(self.gen_measure_bn_ln_4(self.gen_measure_ln_4(measure)), True)
        measure = F.softmax(measure, dim = 1)
        
        return img, measure
        

class netWae(nn.Module):
    def __init__(self, encoder, decoder, ngpu=1):
        super(netWae, self).__init__()
        self.netG = decoder
        self.netD = encoder
        self.ngpu = ngpu 
        
        
    def weight_init_m(self, module, fun, args):
        """
        Initializes weights for the given module and its submodules using the provided
        function and arguments. This function supports verbosity for debugging or
        logging purposes during the initialization process.

        :param module: The root module containing submodules to initialize.
        :param fun: The function used to initialize weights of submodules.
        :param args: The arguments to pass to the weight initialization function.
                     The third element in the tuple indicates whether verbose output
                     is enabled or not.
        """
        verbose = args[2]

        for m in module._modules:
            if verbose:
                print(m)
            for n in module._modules[m]._modules:
                if verbose:
                    print(n)
                fun(module._modules[m]._modules[n], (args[0], args[1]), verbose)

    # weight_init
    def weight_init(self, fun, args_netD, args_netG):
        self.weight_init_m(self.netD, fun, args_netD)
        self.weight_init_m(self.netG, fun, args_netG)

    """
    # weight_init
    def weight_init(self, fun):
        for m in self._modules:
            fun(self._modules[m])
    """
    def forward(self, images, labels):
        output = self.netG(self.netD(images, labels))
        return output 
       
    