# -*- coding: utf-8 -*-
"""
The neural network model for the conditional Variational Autoencoder (cVAE).
It uses the same encoder architecture as jUnet but with a different decoder.
"""

import torch
import torch.nn as nn


class netD(nn.Module):  # encoder
    def __init__(
        self,
        encoder_input_image_channels=1,
        encoder_init_conv_channels=64,
        ngpu=1,
        ndata=81,
        height=50,
        width=40,
        latent_dim=16,
        use_conv_bias=True,
    ):
        super().__init__()
        self.ngpu = ngpu
        self.ndata = ndata
        self.npx = height
        self.npy = width
        self.dz = latent_dim
        self.ndf = encoder_init_conv_channels
        self.nc = encoder_input_image_channels

        self.img_process = nn.Sequential(
            # in: (b, 1, 50, 40)
            nn.ReflectionPad2d(
                2
            ),  # adds 2 pixels to each side of the image. out: (b, 1, 54, 44)
            nn.Conv2d(
                self.nc, self.ndf, 5, 2, 0, bias=use_conv_bias
            ),  # out: (b, 64, 25, 20)
            nn.BatchNorm2d(self.ndf),  # out: (b, 64, 25, 20)
            nn.PReLU(),  # out: (b, 64, 25, 20)
            nn.ReflectionPad2d(2),  # out: (b, 64, 29, 24)
            nn.Conv2d(
                self.ndf, self.ndf * 2, 5, 2, 0, bias=use_conv_bias
            ),  # out: (b, 128, 13, 10)
            nn.BatchNorm2d(self.ndf * 2),  # out: (b, 128, 13, 10)
            nn.PReLU(),  # out: (b, 128, 13, 10)
            nn.Conv2d(
                self.ndf * 2, self.ndf * 4, (5, 2), 1, 0, bias=use_conv_bias
            ),  # out: (b, 256, 9, 9)
            nn.BatchNorm2d(self.ndf * 4),  # out: (b, 256, 9, 9)
            nn.PReLU(),  # out: (b, 256, 9, 9)
            nn.Conv2d(
                self.ndf * 4, self.ndf * 8, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 512, 9, 9)
            nn.BatchNorm2d(self.ndf * 8),  # out: (b, 512, 9, 9)
            nn.PReLU(),  # out: (b, 512, 9, 9)
            nn.Conv2d(
                self.ndf * 8, 1, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 1, 9, 9)
            nn.BatchNorm2d(1),  # out: (b, 1, 9, 9)
            nn.PReLU(),  # out: (b, 1, 9, 9)
        )

        self.measure_process = nn.Sequential(
            # in: (b, 1, 81)
            nn.ReflectionPad1d(1),  # out: (b, 1, 83)
            nn.Conv1d(
                1, self.ndf * 4, 3, 2, 0, bias=use_conv_bias
            ),  # out: (b, 256, 41)
            nn.BatchNorm1d(self.ndf * 4),  # out: (b, 256, 41)
            nn.PReLU(),  # out: (b, 256, 41)
            nn.Conv1d(
                self.ndf * 4, self.ndf * 8, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 512, 41)
            nn.BatchNorm1d(self.ndf * 8),  # out: (b, 512, 41)
            nn.PReLU(),  # out: (b, 512, 41)
            nn.Conv1d(self.ndf * 8, 1, 1, 1, 0, bias=use_conv_bias),  # out: (b, 1, 41)
            nn.BatchNorm1d(1),  # out: (b, 1, 41)
            nn.PReLU(),  # out: (b, 1, 41)
        )

        self.gen_code = nn.Sequential(
            # in: (b, 81 + 41)
            nn.Linear(81 + 41, 81 + 41),  # out: (b, 81 + 41)
            nn.BatchNorm1d(81 + 41),  # out: (b, 81 + 41)
            nn.PReLU(),  # out: (b, 81 + 41)
            nn.Linear(81 + 41, self.dz * 3),  # out: (b, dz)
            nn.BatchNorm1d(self.dz * 3),  # out: (b, dz)
            nn.PReLU(),  # out: (b, dz)
            nn.Linear(self.dz * 3, self.dz * 2),  # out: (b, dz)
            nn.BatchNorm1d(self.dz * 2),  # out: (b, dz)
            nn.PReLU(),  # out: (b, dz)
            # nn.Linear(self.dz * 2, self.dz), # out: (b, dz)
            # nn.PReLU() # out: (b, dz)
        )

        self.fc_mu = nn.Linear(self.dz * 2, self.dz)
        self.fc_logvar = nn.Linear(self.dz * 2, self.dz)

    def forward(self, x, y):
        data = torch.cat([x, y], dim=1)  # concatenate image and measure
        # takes input (b, 2081) : concatenated image (2000 : 50 x 40) and measure (81)
        batch_size = data.shape[0]
        img = (
            data[:, 0 : self.npx * self.npy]
            .view(batch_size, self.npx, self.npy)
            .unsqueeze(1)
        )

        measure = (
            data[:, self.npx * self.npy :].view(batch_size, self.ndata).unsqueeze(1)
        )

        if data.is_cuda and self.ngpu > 1:
            imgcode = nn.parallel.data_parallel(
                self.img_process, img, range(self.ngpu)
            ).squeeze(
                1
            )  # squeeze : remove the channel dimension
            imgcode = imgcode.reshape([batch_size, 81])
            measurecode = nn.parallel.data_parallel(
                self.measure_process, measure, range(self.ngpu)
            )
            measurecode = measurecode.squeeze(1)
            fullcode = torch.cat((imgcode, measurecode), 1)
            output = nn.parallel.data_parallel(
                self.gen_code, fullcode, range(self.ngpu)
            )
            mu = nn.parallel.data_parallel(self.fc_mu, output, range(self.ngpu))
            logvar = nn.parallel.data_parallel(self.fc_logvar, output, range(self.ngpu))

        else:
            imgcode = self.img_process(img).squeeze(
                1
            )  # squeeze : remove the channel dimension
            imgcode = imgcode.reshape([batch_size, 81])
            measurecode = self.measure_process(measure)
            measurecode = measurecode.squeeze(1)
            fullcode = torch.cat((imgcode, measurecode), 1)
            output = self.gen_code(fullcode)
            mu = self.fc_mu(output)
            logvar = self.fc_logvar(output)

        return mu, logvar


class netG(nn.Module):  # decoder/generator
    def __init__(
        self,
        decoder_output_image_channels=1,
        decoder_init_conv_channels=64,
        ngpu=1,
        ndata=81,
        latent_dim=16,
        height=50,
        width=40,
        use_conv_bias=True,
    ):
        super().__init__()
        self.ngpu = ngpu
        self.dz = latent_dim
        self.ndata = ndata
        self.nc = decoder_output_image_channels
        self.ngf = decoder_init_conv_channels
        self.npx = height
        self.npy = width

        self.invert_code = nn.Sequential(
            # in: (b, dz)
            nn.Linear(
                self.dz + self.ndata, self.dz + self.ndata
            ),  # out: (b, dz + ndata)
            nn.BatchNorm1d(self.dz + self.ndata),  # out: (b, dz + ndata)
            nn.PReLU(),  # out: (b, dz + ndata)
            nn.Linear(self.dz + self.ndata, 81),  # out: (b, 81)
            nn.BatchNorm1d(81),  # out: (b, 81)
            nn.PReLU(),  # out: (b, 81)
        )

        self.gen_image = nn.Sequential(
            # in: (b, 1, 9, 9)
            nn.ConvTranspose2d(1, self.ngf * 8, 1, 1, 0, bias=use_conv_bias),
            # out: (b, 512, 9, 9)
            nn.ReflectionPad2d(1),  # out: (b, 512, 11, 11)
            nn.Conv2d(
                self.ngf * 8, self.ngf * 8, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 512, 9, 9)
            nn.BatchNorm2d(self.ngf * 8),  # out: (b, 512, 9, 9)
            nn.PReLU(),  # out: (b, 512, 9, 9)
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=use_conv_bias),
            # out: (b, 256, 9, 9)
            nn.ReflectionPad2d(1),  # out: (b, 256, 11, 11)
            nn.Conv2d(
                self.ngf * 4, self.ngf * 4, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 256, 9, 9)
            nn.BatchNorm2d(self.ngf * 4),  # out: (b, 256, 9, 9)
            nn.PReLU(),  # out: (b, 256, 9, 9)
            nn.ConvTranspose2d(
                self.ngf * 4, self.ngf * 2, (5, 2), 1, 0, bias=use_conv_bias
            ),
            # out: (b, 128, 13, 10)
            nn.ReflectionPad2d(1),  # out: (b, 128, 15, 12)
            nn.Conv2d(
                self.ngf * 2, self.ngf * 2, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 128, 13, 10)
            nn.BatchNorm2d(self.ngf * 2),  # out: (b, 128, 13, 10)
            nn.PReLU(),  # out: (b, 128, 13, 10)
            nn.ConvTranspose2d(
                self.ngf * 2, self.ngf * 2, (5, 6), 2, 2, bias=use_conv_bias
            ),  # out: (b, 128, 25, 20)
            nn.ReflectionPad2d(1),  # out: (b, 128, 27, 22)
            nn.Conv2d(
                self.ngf * 2, self.ngf, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 64, 25, 20)
            nn.BatchNorm2d(self.ngf),  # out: (b, 64, 25, 20)
            nn.PReLU(),  # out: (b, 64, 25, 20)
            nn.ConvTranspose2d(
                self.ngf, self.ngf, 6, 2, 2, bias=use_conv_bias
            ),  # out: (b, 64, 50, 40)
            nn.ReflectionPad2d(1),  # out: (b, 64, 52, 42)
            nn.Conv2d(
                self.ngf, self.nc, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 1, 50, 40)
            nn.ReLU(),  # out: (b, 1, 50, 40) # ReLU is used to ensure the output is positive
        )

    def forward(self, z, y):
        batch_size = z.shape[0]
        if y.shape[0] == 1 and batch_size > 1:
            y = y.expand(batch_size, -1)

        code = torch.cat([z, y], dim=1)

        # takes input (b, dz)
        batch_size = code.shape[0]

        if code.is_cuda and self.ngpu > 1:
            inverted_code = nn.parallel.data_parallel(
                self.invert_code, code, range(self.ngpu)
            )
            img = inverted_code.view(batch_size, 9, 9).unsqueeze(1)
            img = nn.parallel.data_parallel(self.gen_image, img, range(self.ngpu))
            img = img.view(batch_size, self.npx * self.npy)

        else:
            inverted_code = self.invert_code(code)
            img = inverted_code.view(batch_size, 9, 9).unsqueeze(1)
            img = self.gen_image(img)
            img = img.view(batch_size, self.npx * self.npy)
            img = img.view(batch_size, self.npx * self.npy)

        return img


class cVAE(nn.Module):
    def __init__(self, encoder, decoder, ngpu=1):
        super().__init__()
        self.netD = encoder
        self.netG = decoder
        self.ngpu = ngpu
        self.latent_dim = self.netD.dz  # Latent dimension is the same as in the encoder

    # weight_init
    def weight_init_m(self, module, fun, args):
        """
        Function to run through a model modules and initialize its weights
        :param args: A list containing two elements, first is the initialization function to use for the modules,
        the second is whether to print to console modules names as they are being initialized.
        :return:
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

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_samples(self, y, sample_size):
        """Generate samples conditioned on given conditions"""
        self.eval()
        with torch.no_grad():
            # Get the device of the model parameters
            device = next(self.parameters()).device

            # Sample from standard normal distribution
            z = torch.randn(sample_size, self.latent_dim, device=device)
            dtype = z.dtype

            if isinstance(y, torch.Tensor):
                y = y.to(device=device, dtype=dtype)

            # Decode with conditions
            generated = self.netG(z, y)

        return generated

    def forward(self, x, y):
        mu, logvar = self.netD(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.netG(z, y)
        return recon_x, mu, logvar
