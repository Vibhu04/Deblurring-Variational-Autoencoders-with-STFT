<h1 align="center">
  <br>
  Deblurring Variational Autoencoders with Short-Time Fourier Transform 
  <br>
</h1>
  <p align="center">
    <a href="https://github.com/Vibhu04">Vibhu Dalal</a>
  </p>


> **Deblurring Variational Autoencoders with Short-Time Fourier Transform**<br>
> Vibhu Dalal <br>
>
> **Abstract:** *Variational Autoencoders (VAEs) are powerful generative models, however their generated samples are known to suffer from a characteristic blurriness, as compared to the outputs of alternative generating techniques. Extensive research efforts have been made to tackle this problem, and several works have focused on modifying the reconstruction term of the evidence lower bound (ELBO). In particular, many have experimented with augmenting the reconstruction loss with losses in the frequency domain. Such loss functions usually employ the Fourier transform to explicitly penalise the lack of higher frequency components
in the generated samples, which are responsible for sharp visual features. In this paper, we explore the aspects of previous such approaches which aren’t well understood, and we propose an augmentation to the reconstruction term in response to them. Our reasoning leads us to use the short-time Fourier transform and to emphasise on local phase coherence between the input and output samples. We illustrate the potential of our proposed loss on the MNIST dataset by providing both qualitative and quantitative results.*

Access the full paper [here](https://github.com/Vibhu04/Deblurring-Variational-Autoencoders-with-STFT/blob/main/paper.pdf).

## Usage
### Install requirements
After cloning the repository, install the requirements with:

```bash
$ pip install -r requirements.txt
```
### Generate samples
The repository contains 5 checkpoints of a VAE model in `checkpoints/`, which correspond to the following reconstruction loss functions which were tested during the training of the VAE: 
- L1
- L2
- SSIM
- DFT + SSIM
- Ours

Samples can be generated from the models by running `generate.py`. An example run would be:
```
$ python generate.py --loss=ssim, --num_samples=16 --name=gen_samples --out_dir=results
```
The following flags can be specified:
```
$ python generate.py --help
usage: generate.py [-h] [--loss LOSS] [--num_samples NUM_SAMPLES] [--name NAME]
               [--out_dir OUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --loss LOSS           Options: l1, l2, ssim, dft+ssim, ours
  --num_samples NUM_SAMPLES
                        Number of samples to generate
  --name NAME           Name of the generated image
  --out_dir OUT_DIR     Name of output directory
```

### Train the model
To train the VAE model, please refer to the configuration files `config/train_config.yml` and `config/model_config.yml` to customise the training procedure. Once the configuration files are ready, start the training with:

```
$ python train.py 
```

Note: the configuration parameters can be either set from before in the configuration files or they can be set with the previous command, e.g.
```
$ python train.py --batch_size=40 min_lr=0.0001 max_lr=0.001 --epochs=30
```


## Citation
If you use or extend this work, please cite it as below:
```
@software{Vibhu_Dalal_Deblurring-Variational-Autoencoders-with-Short-Time-Fourier-transform_2023,
  author = {Dalal, Vibhu},
  month = {3},
  title = {Deblurring-Variational-Autoencoders-with-Short-Time-Fourier-transform},
  url = {https://github.com/Vibhu04/Deblurring-Variational-Autoencoders-with-STFT},
  version = {1.0.0},
  year = {2023}
}
```
