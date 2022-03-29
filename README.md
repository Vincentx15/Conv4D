# Conv4D : 4D convolutions for tensorflow and jax

For some reason, 4D convolutions are not implemented in major DL frameworks. In this repository, 
I implement an object interface for TF2 using the Sonnet framework and for JAX using the Haiku interface.

This repository extends an [existing tf implementation](https://github.com/funkey/conv4d) with an OOP point of
view along with support for strides and padding. I chose to base myself on the sonnet convolution implementation 
for increased flexibility on the padding function.
I thank the authors of [this excellent pytorch general implementation](https://github.com/pvjosue/pytorch_convNd/blob/master/convNd.py) this implementation is roughly a tf translation.

To use the python scripts, you can simply paste them in your code. Some example usage and timing are provided at the bottom
of the scripts.
On my hardware, both CPU and GPU, the uncompiled runs are faster on TF and compilation and compiled runtimes are much
better on JAX. 