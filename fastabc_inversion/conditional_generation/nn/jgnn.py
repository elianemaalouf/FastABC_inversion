# -*- coding: utf-8 -*-
"""
The neural network model for the joint generative neural network (JGNN).
"""

import torch
import torch.nn as nn


class ConstrainedPReLU(nn.PReLU):
    def __init__(
        self, num_parameters=1, init=0.25, max_value=1.0, device=None, dtype=None
    ):
        super().__init__(
            num_parameters=num_parameters, init=init, device=device, dtype=dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        self.max_value = max_value
        self.register_buffer(
            "max_value_tensor", torch.tensor(max_value, **factory_kwargs)
        )  # Maximum absolute value of 'a' to constrain Lipschitz constant

    def forward(self, x):
        # Constrain 'a' to be within [-max_value, max_value]
        with torch.no_grad():
            self.weight.data.clamp_(min=-self.max_value, max=self.max_value)
        return super().forward(x)


class netD(nn.Module):
    def __init__(
        self,
        input_image_channels=1,
        encoder_channels=128,
        ngpu=1,
        input_label_size=10,
        input_image_height=32,
        input_image_width=32,
        latent_dim=100,
    ):
        super().__init__()
        self.ngpu = ngpu
        self.input_label_size = input_label_size
        self.latent_dim = latent_dim
        self.encoder_channels = encoder_channels
        self.img_channels = input_image_channels
        self.img_height = input_image_height
        self.img_width = input_image_width

        self.img_process = nn.Sequential(
            nn.Conv2d(
                self.img_channels, self.encoder_channels, 4, 2, 1
            ),  # (1, 32, 32) -> (128, 16, 16)
            nn.BatchNorm2d(self.encoder_channels),
            nn.PReLU(),
            nn.Conv2d(
                self.encoder_channels, self.encoder_channels * 2, 4, 2, 1
            ),  # (128, 16, 16) -> (256, 8, 8)
            nn.BatchNorm2d(self.encoder_channels * 2),
            nn.PReLU(),
            nn.Conv2d(
                self.encoder_channels * 2, 1, 2, 2, 4
            ),  # (256, 8, 8) -> (1, 8, 8)
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )

        self.label_process = nn.Sequential(
            nn.Linear(self.input_label_size, self.input_label_size),
            nn.BatchNorm1d(self.input_label_size),
            nn.PReLU(),
        )

        self.final_process = nn.Sequential(
            nn.Linear(74, self.latent_dim * 3),
            nn.BatchNorm1d(self.latent_dim * 3),
            nn.PReLU(),
            nn.Linear(self.latent_dim * 3, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2),
            nn.PReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.PReLU(),
        )

    def forward(self, input_images, input_labels):
        batch_size = input_images.size(0)

        # process image
        img_out = self.img_process(input_images)  # (batch_size, 1, 8, 8)
        # flatten to (batch_size, 64)
        img_out = img_out.view(batch_size, -1)

        # process labels
        label_out = self.label_process(input_labels)  # (batch_size, 10)

        # concatenate image and label features
        combined = torch.cat((img_out, label_out), dim=1)  # (batch_size, 64 + 10)

        # final processing to latent space
        latent_out = self.final_process(combined)  # (batch_size, latent_dim)

        return latent_out


class netG(nn.Module):
    def __init__(
        self,
        output_image_channels=1,
        decoder_channels=128,
        ngpu=1,
        output_label_size=10,
        output_image_height=32,
        output_image_width=32,
        latent_dim=100,
        contrained_prelu=False,
        image_process_type="transposed_conv",
    ):
        super().__init__()
        self.ngpu = ngpu
        self.output_label_size = output_label_size
        self.latent_dim = latent_dim
        self.decoder_channels = decoder_channels
        self.output_image_channels = output_image_channels
        self.output_image_height = output_image_height
        self.output_image_width = output_image_width
        self.activation = ConstrainedPReLU if contrained_prelu else nn.PReLU
        self.image_process_type = image_process_type
        self.output_image_size = (
            self.output_image_channels
            * self.output_image_height
            * self.output_image_width
        )

        # longer Y inversion
        self.invert_label = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2),
            self.activation(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 4),
            nn.BatchNorm1d(self.latent_dim * 4),
            self.activation(),
            nn.Linear(self.latent_dim * 4, self.output_label_size),
            nn.BatchNorm1d(self.output_label_size),
            self.activation(),
            nn.Linear(self.output_label_size, self.output_label_size),
            nn.BatchNorm1d(self.output_label_size),
            self.activation(),
            nn.Linear(self.output_label_size, self.output_label_size),
            nn.Softmax(dim=1),  # output label probabilities
        )
        """
        # shorter Y inversion
        self.invert_label = nn.Sequential(
            nn.Linear(self.latent_dim, self.output_label_size),
            nn.BatchNorm1d(self.output_label_size),
            self.activation(),

            nn.Linear(self.output_label_size, self.output_label_size),
            nn.Softmax(dim=1) # output label probabilities
        ) """

        self.tranposed_conv = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_dim, self.decoder_channels * 2, 4, 1, 0
            ),  # (latent_dim, 1, 1) -> (256, 4, 4)
            nn.BatchNorm2d(self.decoder_channels * 2),
            self.activation(),
            nn.ConvTranspose2d(
                self.decoder_channels * 2, self.decoder_channels * 2, 6, 2, 2
            ),  # (256, 4, 4) -> (256, 8, 8)
            nn.BatchNorm2d(self.decoder_channels * 2),
            self.activation(),
            nn.ConvTranspose2d(
                self.decoder_channels * 2, self.decoder_channels, 6, 2, 2
            ),  # (256, 8, 8) -> (128, 16, 16)
            nn.BatchNorm2d(self.decoder_channels),
            self.activation(),
            nn.ConvTranspose2d(
                self.decoder_channels, self.output_image_channels, 6, 2, 2
            ),  # (128, 16, 16) -> (1, 32, 32)
            nn.Tanh(),  # output image pixel values in range [-1, 1]
        )

        self.linear_image = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),  # First expansion
            nn.BatchNorm1d(self.latent_dim * 2),
            self.activation(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 4),  # Second expansion
            nn.BatchNorm1d(self.latent_dim * 4),
            self.activation(),
            nn.Linear(self.latent_dim * 4, self.latent_dim * 8),  # Start reducing
            nn.BatchNorm1d(self.latent_dim * 8),
            self.activation(),
            nn.Linear(self.latent_dim * 8, self.output_image_size),  # Continue reducing
            nn.BatchNorm1d(self.output_image_size),
            self.activation(),
            nn.Linear(
                self.output_image_size, self.output_image_size
            ),  # Final layer to image size
            nn.Tanh(),  # output image pixel values in range [-1, 1]
        )

        # upsample + conv
        # First layer: Linear projection to initial spatial dimensions
        self.initial_projection = nn.Sequential(
            nn.Linear(
                self.latent_dim, self.decoder_channels * 2 * 4 * 4
            ),  # (latent_dim,) -> (256 * 16,)
            nn.BatchNorm1d(self.decoder_channels * 2 * 4 * 4),
            self.activation(),
        )

        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # (256, 4, 4) -> (256, 8, 8)
            nn.Conv2d(self.decoder_channels * 2, self.decoder_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(self.decoder_channels * 2),
            self.activation(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # (256, 8, 8) -> (256, 16, 16)
            nn.Conv2d(self.decoder_channels * 2, self.decoder_channels, 3, 1, 1),
            nn.BatchNorm2d(self.decoder_channels),
            self.activation(),
            nn.Upsample(
                scale_factor=2, mode="nearest"
            ),  # (128, 16, 16) -> (128, 32, 32)
            nn.Conv2d(self.decoder_channels, self.output_image_channels, 3, 1, 1),
            nn.Tanh(),
        )

        if self.image_process_type == "transposed_conv":
            self.invert_image = self.tranposed_conv
        elif self.image_process_type == "linear":
            self.invert_image = self.linear_image
        elif self.image_process_type == "upsample_conv":
            self.invert_image = self.upsample_conv
        else:
            raise ValueError(
                f"Unknown image_process_type: {image_process_type}, use one of ['transposed_conv', 'linear', 'upsample_conv']"
            )

    def forward(self, latent_vector):
        batch_size = latent_vector.size(0)

        # invert labels
        label_out = self.invert_label(latent_vector)  # (batch_size, output_label_size)

        # invert image
        if self.image_process_type == "transposed_conv":
            img_latent = latent_vector.view(
                batch_size, self.latent_dim, 1, 1
            )  # reshape to (batch_size, latent_dim, 1, 1)

        if self.image_process_type == "upsample_conv":
            img_latent = self.initial_projection(latent_vector).view(
                batch_size, self.decoder_channels * 2, 4, 4
            )

        if self.image_process_type == "linear":
            img_latent = latent_vector  # keep as is for linear processing

        image_out = self.invert_image(
            img_latent
        )  # (batch_size, output_image_channels, output_image_height, output_image_width)

        if self.image_process_type == "linear":
            image_out = image_out.view(
                batch_size,
                self.output_image_channels,
                self.output_image_height,
                self.output_image_width,
            )

        return image_out, label_out


class jGNN(nn.Module):
    def __init__(self, encoder, decoder, ngpu=1):
        super().__init__()
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

    def forward(self, images, labels):
        output = self.netG(self.netD(images, labels))
        return output
