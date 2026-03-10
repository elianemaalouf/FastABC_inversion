# -*- coding: utf-8 -*-
"""
The neural network model for the joint U-Net model. There are no skip connections between the encoders and decoders.
It is made of two U-Net models, one for a 2D image (1 channel) and another for a 1D signal (1 channel).
The 2D U-Net model is used to process the 2D image and the 1D U-Net model is used to process the 1D signal.
After the encoding of each model, the encoded features are concatenated and processed by fully connected layers.
The output of the fully connected layers is then processed by two decoders, one for the 2D image and another for the 1D signal.
"""

import torch
import torch.nn as nn
from fastabc_inversion.geo_problems.nn.activations import (
    BetaSwish, ConstrainedBetaSwish, ConstrainedPReLU)

_AVAILABLE_ACTIVATIONS = [
    "relu",
    "leaky_relu",
    "prelu",
    "swish",
    "silu",
    "beta_swish",
    "constrained_beta_swish",
    "constrained_prelu",
]


def get_activation_fn(name, activation_kwargs=None, channels=None):
    name = name.lower()
    if name not in _AVAILABLE_ACTIVATIONS:
        raise ValueError(
            f"Activation function {name} not available. Choose one of {_AVAILABLE_ACTIVATIONS}"
        )

    if activation_kwargs is not None:
        if len(activation_kwargs) == 0:
            activation_kwargs = None
        else:
            if (
                "num_parameters" in activation_kwargs
                and activation_kwargs["num_parameters"] == -1
            ):
                activation_kwargs["num_parameters"] = channels

    activation_instance = None
    if name == "relu":
        activation_instance = (
            nn.ReLU() if activation_kwargs is None else nn.ReLU(**activation_kwargs)
        )
    elif name == "leaky_relu":
        activation_instance = (
            nn.LeakyReLU()
            if activation_kwargs is None
            else nn.LeakyReLU(**activation_kwargs)
        )
    elif name == "prelu":
        activation_instance = (
            nn.PReLU() if activation_kwargs is None else nn.PReLU(**activation_kwargs)
        )
    elif name in ["swish", "silu"]:
        activation_instance = (
            nn.SiLU() if activation_kwargs is None else nn.SiLU(**activation_kwargs)
        )
    elif name == "beta_swish":
        activation_instance = (
            BetaSwish() if activation_kwargs is None else BetaSwish(**activation_kwargs)
        )  # custom
    elif name == "constrained_beta_swish":
        activation_instance = (
            ConstrainedBetaSwish()
            if activation_kwargs is None
            else ConstrainedBetaSwish(**activation_kwargs)
        )  # custom
    elif name == "constrained_prelu":
        activation_instance = (
            ConstrainedPReLU()
            if activation_kwargs is None
            else ConstrainedPReLU(**activation_kwargs)
        )  # custom

    return activation_instance


class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


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
        activation_dict_encoder=None,
    ):
        super().__init__()
        self.ngpu = ngpu
        self.ndata = ndata
        self.npx = height
        self.npy = width
        self.dz = latent_dim
        self.ndf = encoder_init_conv_channels
        self.nc = encoder_input_image_channels
        activation_name = activation_dict_encoder["name"]
        activation_kwargs = {
            k: v for k, v in activation_dict_encoder.items() if k != "name"
        }

        self.img_process = nn.Sequential(
            # in: (b, 1, 50, 40)
            nn.ReflectionPad2d(
                2
            ),  # adds 2 pixels to each side of the image. out: (b, 1, 54, 44)
            nn.Conv2d(
                self.nc, self.ndf, 5, 2, 0, bias=use_conv_bias
            ),  # out: (b, 64, 25, 20)
            nn.BatchNorm2d(self.ndf),  # out: (b, 64, 25, 20)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ndf
            ),  # out: (b, 64, 25, 20)
            # nn.PReLU(), # out: (b, 64, 25, 20)
            nn.ReflectionPad2d(2),  # out: (b, 64, 29, 24)
            nn.Conv2d(
                self.ndf, self.ndf * 2, 5, 2, 0, bias=use_conv_bias
            ),  # out: (b, 128, 13, 10)
            nn.BatchNorm2d(self.ndf * 2),  # out: (b, 128, 13, 10)
            get_activation_fn(
                activation_name, activation_kwargs, channels=(self.ndf * 2)
            ),  # out: (b, 128, 13, 10)
            # nn.PReLU(), # out: (b, 128, 13, 10)
            nn.Conv2d(
                self.ndf * 2, self.ndf * 4, (5, 2), 1, 0, bias=use_conv_bias
            ),  # out: (b, 256, 9, 9)
            nn.BatchNorm2d(self.ndf * 4),  # out: (b, 256, 9, 9)
            get_activation_fn(
                activation_name, activation_kwargs, channels=(self.ndf * 4)
            ),  # out: (b, 256, 9, 9)
            # nn.PReLU(), # out: (b, 256, 9, 9)
            nn.Conv2d(
                self.ndf * 4, self.ndf * 8, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 512, 9, 9)
            nn.BatchNorm2d(self.ndf * 8),  # out: (b, 512, 9, 9)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ndf * 8
            ),  # out: (b, 512, 9, 9)
            # nn.PReLU(), # out: (b, 512, 9, 9)
            nn.Conv2d(
                self.ndf * 8, 1, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 1, 9, 9)
            nn.BatchNorm2d(1),  # out: (b, 1, 9, 9)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, 1, 9, 9)
            # nn.PReLU(), # out: (b, 1, 9, 9)
        )

        self.measure_process = nn.Sequential(
            # in: (b, 1, 81)
            nn.ReflectionPad1d(1),  # out: (b, 1, 83)
            nn.Conv1d(
                1, self.ndf * 4, 3, 2, 0, bias=use_conv_bias
            ),  # out: (b, 256, 41)
            nn.BatchNorm1d(self.ndf * 4),  # out: (b, 256, 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ndf * 4
            ),  # out: (b, 256, 41)
            # nn.PReLU(), # out: (b, 256, 41)
            nn.Conv1d(
                self.ndf * 4, self.ndf * 8, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 512, 41)
            nn.BatchNorm1d(self.ndf * 8),  # out: (b, 512, 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ndf * 8
            ),  # out: (b, 512, 41)
            # nn.PReLU(), # out: (b, 512, 41)
            nn.Conv1d(self.ndf * 8, 1, 1, 1, 0, bias=use_conv_bias),  # out: (b, 1, 41)
            nn.BatchNorm1d(1),  # out: (b, 1, 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, 1, 41)
            # nn.PReLU(), # out: (b, 1, 41)
        )

        self.gen_code = nn.Sequential(
            # in: (b, 81 + 41)
            nn.Linear(81 + 41, 81 + 41),  # out: (b, 81 + 41)
            nn.BatchNorm1d(81 + 41),  # out: (b, 81 + 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, 81 + 41)
            # nn.PReLU(), # out: (b, 81 + 41)
            nn.Linear(81 + 41, self.dz * 3),  # out: (b, dz* 3)
            nn.BatchNorm1d(self.dz * 3),  # out: (b, dz* 3)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, dz* 3)
            # nn.PReLU(), # out: (b, dz* 3)
            nn.Linear(self.dz * 3, self.dz * 2),  # out: (b, dz* 2)
            nn.BatchNorm1d(self.dz * 2),  # out: (b, dz* 2)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, dz* 2)
            # nn.PReLU(), # out: (b, dz* 2)
            nn.Linear(self.dz * 2, self.dz),  # out: (b, dz)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, dz)
            # nn.PReLU() # out: (b, dz)
        )

    def forward(self, data):
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

        else:
            imgcode = self.img_process(img).squeeze(
                1
            )  # squeeze : remove the channel dimension
            imgcode = imgcode.reshape([batch_size, 81])
            measurecode = self.measure_process(measure)
            measurecode = measurecode.squeeze(1)
            fullcode = torch.cat((imgcode, measurecode), 1)
            output = self.gen_code(fullcode)

        return (output, imgcode, measurecode)


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
        activation_dict_decoder=None,
    ):
        super().__init__()
        self.ngpu = ngpu
        self.dz = latent_dim
        self.ndata = ndata
        self.nc = decoder_output_image_channels
        self.ngf = decoder_init_conv_channels
        self.npx = height
        self.npy = width
        activation_name = activation_dict_decoder["name"]
        activation_kwargs = {
            k: v for k, v in activation_dict_decoder.items() if k != "name"
        }

        self.invert_code = nn.Sequential(
            # in: (b, dz)
            nn.Linear(self.dz, self.dz * 2),  # out: (b, dz* 2)
            nn.BatchNorm1d(self.dz * 2),  # out: (b, dz* 2)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, dz* 2)
            # nn.PReLU(), # out: (b, dz* 2)
            nn.Linear(self.dz * 2, self.dz * 3),  # out: (b, dz* 3)
            nn.BatchNorm1d(self.dz * 3),  # out: (b, dz* 3)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, dz* 3)
            # nn.PReLU(), # out: (b, dz* 3)
            nn.Linear(self.dz * 3, 81 + 41),  # out: (b, 81 + 41)
            nn.BatchNorm1d(81 + 41),  # out: (b, 81 + 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, 81 + 41)
            # nn.PReLU(), # out: (b, 81 + 41)
            nn.Linear(81 + 41, 81 + 41),  # out: (b, 81 + 41)
            nn.BatchNorm1d(81 + 41),  # out: (b, 81 + 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=1
            ),  # out: (b, 81 + 41)
            # nn.PReLU() # out: (b, 81 + 41)
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
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ngf * 8
            ),  # out: (b, 512, 9, 9)
            # nn.PReLU(), # out: (b, 512, 9, 9)
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=use_conv_bias),
            # out: (b, 256, 9, 9)
            nn.ReflectionPad2d(1),  # out: (b, 256, 11, 11)
            nn.Conv2d(
                self.ngf * 4, self.ngf * 4, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 256, 9, 9)
            nn.BatchNorm2d(self.ngf * 4),  # out: (b, 256, 9, 9)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ngf * 4
            ),  # out: (b, 256, 9, 9)
            # nn.PReLU(), # out: (b, 256, 9, 9)
            nn.ConvTranspose2d(
                self.ngf * 4, self.ngf * 2, (5, 2), 1, 0, bias=use_conv_bias
            ),
            # out: (b, 128, 13, 10)
            nn.ReflectionPad2d(1),  # out: (b, 128, 15, 12)
            nn.Conv2d(
                self.ngf * 2, self.ngf * 2, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 128, 13, 10)
            nn.BatchNorm2d(self.ngf * 2),  # out: (b, 128, 13, 10)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ngf * 2
            ),  # out: (b, 128, 13, 10)
            # nn.PReLU(), # out: (b, 128, 13, 10)
            nn.ConvTranspose2d(
                self.ngf * 2, self.ngf * 2, (5, 6), 2, 2, bias=use_conv_bias
            ),  # out: (b, 128, 25, 20)
            nn.ReflectionPad2d(1),  # out: (b, 128, 27, 22)
            nn.Conv2d(
                self.ngf * 2, self.ngf, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 64, 25, 20)
            nn.BatchNorm2d(self.ngf),  # out: (b, 64, 25, 20)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ngf
            ),  # out: (b, 64, 25, 20)
            # nn.PReLU(), # out: (b, 64, 25, 20)
            nn.ConvTranspose2d(
                self.ngf, self.ngf, 6, 2, 2, bias=use_conv_bias
            ),  # out: (b, 64, 50, 40)
            nn.ReflectionPad2d(1),  # out: (b, 64, 52, 42)
            nn.Conv2d(
                self.ngf, self.nc, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 1, 50, 40)
            # get_activation_fn(activation_name, activation_kwargs, channels=self.nc), # out: (b, 1, 50, 40)
            nn.ReLU(),  # out: (b, 1, 50, 40) # ReLU is used to ensure the output is positive
        )

        self.gen_measure = nn.Sequential(
            # in: (b, 1, 41)
            nn.ConvTranspose1d(
                1, self.ngf * 8, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 512, 41)
            nn.ReflectionPad1d(1),  # out: (b, 512, 43)
            nn.Conv1d(
                self.ngf * 8, self.ngf * 8, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 512, 41)
            nn.BatchNorm1d(self.ngf * 8),  # out: (b, 512, 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ngf * 8
            ),  # out: (b, 512, 41)
            # nn.PReLU(), # out: (b, 512, 41)
            nn.ConvTranspose1d(
                self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=use_conv_bias
            ),  # out: (b, 256, 41)
            nn.ReflectionPad1d(1),  # out: (b, 256, 43)
            nn.Conv1d(
                self.ngf * 4, self.ngf * 4, 3, 1, 0, bias=use_conv_bias
            ),  # out: (b, 256, 41)
            nn.BatchNorm1d(self.ngf * 4),  # out: (b, 256, 41)
            get_activation_fn(
                activation_name, activation_kwargs, channels=self.ngf * 4
            ),  # out: (b, 256, 41)
            # nn.PReLU(), # out: (b, 256, 41)
            nn.ConvTranspose1d(
                self.ngf * 4, 1, 3, 2, 1, bias=use_conv_bias
            ),  # out: (b, 1, 81)
            nn.ReflectionPad1d(1),  # out: (b, 1, 83)
            nn.Conv1d(1, 1, 3, 1, 0, bias=use_conv_bias),  # out: (b, 1, 81)
            # get_activation_fn(activation_name, activation_kwargs, channels=1),  # out: (b, 1, 81)
            nn.ReLU(),  # out: (b, 1, 81) # ReLU is used to ensure the output is positive
        )

    def forward(self, code):
        # takes input (b, dz)
        batch_size = code.shape[0]

        if code.is_cuda and self.ngpu > 1:
            inverted_code = nn.parallel.data_parallel(
                self.invert_code, code, range(self.ngpu)
            )
            img = inverted_code[:, 0 : self.ndata].view(batch_size, 9, 9).unsqueeze(1)
            img = nn.parallel.data_parallel(self.gen_image, img, range(self.ngpu))
            img = img.view(batch_size, self.npx * self.npy)

            measure = inverted_code[:, self.ndata :].view(batch_size, 41).unsqueeze(1)
            measure = nn.parallel.data_parallel(
                self.gen_measure, measure, range(self.ngpu)
            )
            measure = measure.view(batch_size, self.ndata)

        else:
            inverted_code = self.invert_code(code)
            img = inverted_code[:, 0 : self.ndata].view(batch_size, 9, 9).unsqueeze(1)
            img = self.gen_image(img)
            img = img.view(batch_size, self.npx * self.npy)

            measure = inverted_code[:, self.ndata :].view(batch_size, 41).unsqueeze(1)
            measure = self.gen_measure(measure)
            measure = measure.view(batch_size, self.ndata)

        return torch.cat((img, measure), 1)


class netWae(nn.Module):
    def __init__(self, encoder, decoder, ngpu=1):
        super().__init__()
        self.netG = decoder
        self.netD = encoder
        self.ngpu = ngpu

        """
        self.stacked = nn.Sequential(
            self.netD, 
            self.netG
        )"""

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

    def forward(self, data):
        output = self.netG(self.netD(data)[0])
        # output = self.stacked(data)
        return output
