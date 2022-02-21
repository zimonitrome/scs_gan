# SCSGAN
Generative Adversarial Network using Sharpened Cosine Similarity

This repo is a modified copy of PyTorch's [DCGAN (Deep Convolutional GAN) tutorial](https://github.com/pytorch/examples/tree/master/dcgan) but using [Sharpened Cosine Similarity](https://e2eml.school/scs.html) instead of convolutions.

The concept of SCS is inspired by [Brandon Rohrer](https://github.com/brohrer) ([Main SCS repo](https://github.com/brohrer/sharpened_cosine_similarity_torch)).

The implementation of SCS in this repo is taken from [Lucas Nestler](https://gist.github.com/ClashLuke) ([Gist](https://gist.github.com/ClashLuke/8f6521deef64789e76334f1b72a70d80)).

## Tests

Using the MNIST dataset

### Original DGCAN

Parameters in G: 3,574,656

Parameters in D: 2,763,520

https://user-images.githubusercontent.com/17656709/155038245-4837e3c2-34f6-486e-92f7-e3f791f8c48e.mp4


### DC G + SCS D

Parameters in G: 3,574,656

Parameters in D:   196,143

https://user-images.githubusercontent.com/17656709/155038258-e44f55ed-43e4-4519-899c-6c734f7699c0.mp4


### SCS G + SCS D

Parameters in G: 2,046,987

Parameters in D:   196,143

https://user-images.githubusercontent.com/17656709/155038275-43d9758b-5d0c-4041-bb2e-d89ea5a31ed3.mp4


## TODO

- [ ] Test new SCS implementation by Lucas Nestler without last part of SCS

- [ ] Compare FID scores

- [ ] Compare training time

- [ ] Better SCS Generator without ReLU

- [ ] Try more datasets
