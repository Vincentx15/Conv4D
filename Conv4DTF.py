#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import numpy as np

import sonnet as snt
import sonnet.src.conv as snt_conv
import tensorflow as tf


class Conv4D(snt.Module):
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 stride=(1, 1, 1, 1),
                 name=None,
                 padding='valid',
                 data_format='NCLDHW'):
        """

        Constructs a `Conv4D` module.

        :param output_channels: The number of output channels.
        :param kernel_shape: A 4-tuple
        :param stride:
        :param name: Name of the module.
        :param padding: either 'valid' or 'same' or a tuple of the form ((3, 0), (0, 0), (0, 0), (0, 0)), which follows
        Pytorch. Each inner tuple represents the padding to use before and after for each dim.
        :param data_format: one of NLDHWC or NCLDHW, equivalently channel_first/last. The input will be expected
        to follow this and the output will also respect this ordering.
        """
        super().__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride

        # We implement a padding that also allows for numerical padding.
        # The format is a bit specific and the input one for sonnet is even weirder
        if not isinstance(padding, str):
            self.padding_value, inner_convs_padding = padding[0], padding[1:4]
            self.padding = 'float'
            inner_convs_padding = [lambda i: x for x in inner_convs_padding]
        else:
            assert padding in {'valid', 'same'}
            self.padding = padding
            inner_convs_padding = padding

        l_k, d_k, h_k, w_k = self.kernel_shape
        l_s, d_s, h_s, w_s = self.stride
        self.conv3dlist = [snt_conv.Conv3D(output_channels,
                                               kernel_shape=(d_k, h_k, w_k),
                                               stride=(d_s, h_s, w_s),
                                               data_format='NDHWC',
                                               padding=inner_convs_padding) for _ in range(l_k)]
        self.data_format = data_format

    def __call__(self, inputs):
        """
        Applies the defined convolution to the inputs.

        This will respect the channel_first/last order as specified by the user.
        """
        if isinstance(inputs, np.ndarray):
            inputs_shape = inputs.shape
        else:
            inputs_shape = inputs.get_shape().as_list()
        if self.data_format == 'NLDHWC':
            _, l_i, _, _, _, _ = inputs_shape
        elif self.data_format == 'NCLDHW':
            _, _, l_i, _, _, _ = inputs_shape
        else:
            raise ValueError(f'{self.data_format} is not a valid data format for now')
        l_k = self.kernel_shape[0]
        l_s = self.stride[0]
        # output size for 'valid' convolution
        p_before = 0
        p_after = 0
        if self.padding == 'valid':
            l_o = (l_i - l_k) // l_s + 1
        elif self.padding == 'same':
            assert l_s == 1, "Cannot use same padding with a stride greater than one !"
            l_o = l_i
        elif self.padding == 'float':
            p_before, p_after = self.padding_value
            l_o = (l_i + p_before + p_after - l_k) // l_s + 1
        else:
            raise ValueError

        # output tensors for each 3D frame
        frame_results = [None] * l_o
        at_least_one_output = False

        # convolve each kernel frame i with each input frame j
        for i_kernel in range(l_k):  # kernel
            for j_input in range(l_i):  # input
                # Compute where our frame would fall in the image.
                # We use a kernel offset because the first kernels are supposedly centered.
                # In case of an even kernel size, the mathematical formulation would yield a result for l_o=0.5
                # We choose to remove it : shift the results by -0.5

                # out_frame_padded is centered around the padded input coordinates.
                # It is python based (starts at 0) and accounts for shifting the kernel (applying it as centered)
                # Then out_frame_idx stores the region that is in the valid domain (only full convs are kept)
                kernel_offset = (l_k // 2) - (1 - l_k % 2)
                out_frame_padded = j_input - i_kernel + p_before + kernel_offset
                out_frame_idx = out_frame_padded - (l_k - 1) // 2

                # Only keep the ones with appropriate strides that fall into our image
                if out_frame_idx % l_s:
                    continue
                out_frame_idx = out_frame_idx // l_s
                if out_frame_idx < 0 or out_frame_idx >= l_o:
                    continue

                # Then because the cpu implementation is faulty, we need to translate into channels last formulation
                if self.data_format == 'NCLDHW':
                    selected_inputs = inputs[:, :, j_input, ...]
                    channel_last_inputs = tf.transpose(selected_inputs, perm=[0, 2, 3, 4, 1])
                else:
                    channel_last_inputs = inputs[:, j_input, ...]
                frame_conv3d = self.conv3dlist[i_kernel](channel_last_inputs)
                at_least_one_output = True
                outconv_shape = frame_conv3d.shape
                if frame_results[out_frame_idx] is None:
                    frame_results[out_frame_idx] = frame_conv3d
                else:
                    frame_results[out_frame_idx] += frame_conv3d

        # With insufficient padding, all frames could become border frames could be empty
        if not at_least_one_output:
            raise ValueError('The combination of parameters used in 4D cNNs has yield an empty tensor')
        # With excessive padding, some frames are zero
        for i, frame in enumerate(frame_results):
            if frame is None:
                frame_results[i] = tf.zeros(shape=outconv_shape)

        output = tf.stack(frame_results, axis=1)
        if self.data_format == 'NCLDHW':
            output = tf.transpose(output, perm=[0, 5, 1, 2, 3, 4])
        return output


if __name__ == '__main__':
    pass
    import time

    decoy_input = tf.zeros(shape=(1, 5, 16, 17, 18, 1))
    conv4d_1 = Conv4D(output_channels=8, kernel_shape=(4, 3, 3, 3), stride=(2, 1, 1, 1),
                      data_format='NLDHWC',
                      padding=((3, 0), (0, 0), (0, 0), (0, 0)))

    n_iter = 20
    # First let's look at the uncompiled performance. We will look at it again, because the GPU needs to heat.
    t_0 = time.perf_counter()
    for _ in range(n_iter):
        out = conv4d_1(decoy_input)
    print('Time with an uncompiled model', time.perf_counter() - t_0)


    @tf.function
    def my_forward_apply(x):
        return conv4d_1(x)


    # Now let us compile our function
    t_0 = time.perf_counter()
    out = my_forward_apply(x=decoy_input)
    print('Time to compile', time.perf_counter() - t_0)

    # The compiled timing should be better
    t_0 = time.perf_counter()
    for _ in range(n_iter):
        out = my_forward_apply(x=decoy_input)
    print('Time with a compiled model', time.perf_counter() - t_0)

    # For a fairer comparison, we redo a run as this should be faster than the first one.
    t_0 = time.perf_counter()
    for _ in range(n_iter):
        out = conv4d_1(decoy_input)
    print('Second time with a uncompiled model', time.perf_counter() - t_0)
    print(out.shape)
