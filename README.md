# SCSGAN
Generative Adversarial Network using Sharpened Cosine Similarity

This repo is a modified copy of PyTorch's [DCGAN (Deep Convolutional GAN) tutorial](https://github.com/pytorch/examples/tree/master/dcgan) but using [Sharpened Cosine Similarity](https://e2eml.school/scs.html) instead of convolutions.

The concept of SCS is inspired by [Brandon Rohrer](https://github.com/brohrer) ([Main SCS repo](https://github.com/brohrer/sharpened_cosine_similarity_torch)).

The implementation of SCS in this repo is taken from [Lucas Nestler](https://gist.github.com/ClashLuke) ([Gist](https://gist.github.com/ClashLuke/8f6521deef64789e76334f1b72a70d80)).

## Tests

Using the MNIST dataset

### FID + time

Currently SCS does not seem to be much better than deep conv GANs... More tinkering needed.

This is FID calculated on batches of 64 vs 64 images during training of 25 epochs and ndf (controlling feature maps in D) set to 16.

The lines are actually the average of 3 runs with the seeds 1, 2, 3.

![](media/FID_results.png)

### Videos from the runs

### Vanilla DGCAN

```
Parameters in G: 3,574,656
Parameters in D:   174,784
```

https://user-images.githubusercontent.com/17656709/155624893-cbc3c8d4-21d4-4976-928a-1572859076d7.mp4


### SCS GAN

```
Parameters in G: 3,572,741
Parameters in D:   173,051
```

https://user-images.githubusercontent.com/17656709/155624917-ded582e3-489c-4249-8e72-f136b5f2c222.mp4


## TODO

- [ ] Make SCS Generator better (find replacement for LeakyReLU?)
- [ ] Try more datasets
- [ ] Try smaller models without P
- [x] Test new SCS implementation by Lucas Nestler
- [X] Make p optional in SCS
- [X] Compare FID scores
- [X] Compare training time
- [x] Better SCS Generator without ReLU and normal conv
- [x] Update videos to use universally supported codec
- [X] Update all videos to show new architectures
