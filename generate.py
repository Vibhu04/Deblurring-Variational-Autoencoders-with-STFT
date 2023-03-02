import argparse
import tensorflow as tf

from vae.vae1 import VAE1
from utils import generate_images


def main():

    tf.get_logger().setLevel('WARNING')
    parser = arg_parser()
    args, unknown = parser.parse_known_args()

    loss = args.loss
    losses = ['l1', 'l2', 'ssim', 'dft+ssim', 'ours']

    if loss not in losses:
        raise Exception("The chosen loss function wasn't one of the following: l1, l2, ssim, dft+ssim, ours.")

    model_dir = 'checkpoints/' + loss + '/' + loss
    out_dir = args.out_dir + '/' + loss
    model = VAE1()
    model.load_weights(model_dir)
    random_sample = tf.random.normal(shape=(args.num_samples, model.latent_dim))

    generate_images(model, random_sample, out_dir, args.name)




def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', type=str, default='ours', help="Options: l1, l2, ssim, dft+ssim, ours")
    parser.add_argument('--num_samples', type=int, default=16, help="Number of samples to generate")
    parser.add_argument('--name', type=str, default='user_gen', help="Name of the generated image")
    parser.add_argument('--out_dir', type=str, default='results', help="Name of output directory")

    return parser



if __name__ == '__main__':
    main()

