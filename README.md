<h1 align="center">
  <br>
  Deblurring Variational Autoencoders with short-time Fourier transform 
  <br>
</h1>
  <p align="center">
    <a href="https://github.com/Vibhu04">Vibhu Dalal</a>
  </p>


> **Deblurring Variational Autoencoders with short-time Fourier transform**<br>
> Vibhu Dalal <br>
>
> **Abstract:** *Variational Autoencoders (VAEs) are powerful generative models, however their generated samples are known to suffer from a characteristic blurriness, as compared to the outputs of alternative generating techniques. Extensive research efforts have been made to tackle this problem, and several works have focused on modifying the reconstruction term of the evidence lower bound (ELBO). In particular, many have experimented with augmenting the reconstruction loss with losses in the frequency domain. Such loss functions usually employ the Fourier transform to explicitly penalise the lack of higher frequency components
in the generated samples, which are responsible for sharp visual features. In this paper, we explore the aspects of previous such approaches which aren’t well understood, and we propose an augmentation to the reconstruction term in response to them. Our reasoning leads us to use the short-time Fourier transform and to emphasise on local phase coherence between the input and output samples. We illustrate the potential of our proposed loss on the MNIST dataset by providing both qualitative and quantitative results.*

## Citation
```
@software{Vibhu_Dalal_Deblurring-Variational-Autoencoders-with-short-time-Fourier-transform_2023,
  author = {Dalal, Vibhu},
  month = {3},
  title = {Deblurring-Variational-Autoencoders-with-short-time-Fourier-transform},
  url = {https://github.com/Vibhu04/Deblurring-Variational-Autoencoders-with-STFT},
  version = {1.0.0},
  year = {2023}
}
```

## To run the demo

To run the demo, you will need to have a CUDA capable GPU, PyTorch >= v1.3.1 and cuda/cuDNN drivers installed.
Install the required packages:

    pip install -r requirements.txt
  
Download pre-trained models:

    python training_artifacts/download_all.py

Run the demo:

    python interactive_demo.py

You can specify **yaml** config to use. Configs are located here: https://github.com/podgorskiy/ALAE/tree/master/configs.
By default, it uses one for FFHQ dataset.
You can change the config using `-c` parameter. To run on `celeb-hq` in 256x256 resolution, run:

    python interactive_demo.py -c celeba-hq256

However, for configs other then FFHQ, you need to obtain new principal direction vectors for the attributes.

## Repository organization

#### Running scripts

The code in the repository is organized in such a way that all scripts must be run from the root of the repository.
If you use an IDE (e.g. PyCharm or Visual Studio Code), just set *Working Directory* to point to the root of the repository.

If you want to run from the command line, then you also need to set **PYTHONPATH** variable to point to the root of the repository.

For example, let's say we've cloned repository to *~/ALAE* directory, then do:

    $ cd ~/ALAE
    $ export PYTHONPATH=$PYTHONPATH:$(pwd)

![pythonpath](https://podgorskiy.com/static/pythonpath.svg)

Now you can run scripts as follows:

    $ python style_mixing/stylemix.py
