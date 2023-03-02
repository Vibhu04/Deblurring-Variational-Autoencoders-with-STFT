from tensorflow_addons.optimizers import CyclicalLearningRate
import argparse
import yaml
import tensorflow as tf

from vae.vae1 import VAE1
from vae.vae2 import VAE2
from utils import *



def main():

    parser_train, parser_model = arg_parser()
    train_args, unknown = parser_train.parse_known_args()
    model_args, unknown = parser_model.parse_known_args()

    verify_dirs(train_args)
    train_summary_writer, test_summary_writer = create_writers(train_args)
    train_dataset, test_dataset = get_datasets(train_args, model_args)

    if train_args.dataset == 'mnist':
        model = VAE1(model_args)
    elif train_args.dataset == 'cartoons':
        model = VAE2(model_args)

    if train_args.load_model:
        model.load_weights(train_args.model_dir)
        print("Model loaded.")

    cyclical_learning_rate = CyclicalLearningRate(
        initial_learning_rate=train_args.min_lr,
        maximal_learning_rate=train_args.max_lr,
        step_size=train_args.lr_step_size,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        scale_mode='cycle')

    optimizer = tf.keras.optimizers.Adam(learning_rate=cyclical_learning_rate)

    if test_dataset is not None:
        random_sample = tf.random.normal(shape=(train_args.num_samples, model.latent_dim))
        test_set = list(test_dataset.as_numpy_iterator())
        test_sample = test_set[0][:8]

    iters = 0

    # Training loop
    for epoch in range(train_args.epochs):
        print("\nStart of epoch %d" % (epoch,))
        for step, (x_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                _, losses = model(x_batch_train)

            loss = losses["loss"]
            grads = tape.gradient(loss, model.trainable_weights)

            print('Epoch: {}, batch: {}, loss: {}'.format(epoch, step, loss))

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if iters % train_args.log_freq == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('train_loss', loss, step=iters)
                    tf.summary.scalar('kld_loss', losses["kld_loss"], step=iters)
                    tf.summary.scalar('recons_loss', losses["recons_loss"], step=iters)
                    tf.summary.scalar('stft_loss', losses["stft_loss"], step=iters)
                    tf.summary.scalar('learning rate', optimizer.lr.numpy(), step=iters)


            if iters % train_args.test_freq == 0:
                if test_dataset is not None:
                    test_loss = tf.keras.metrics.Mean()
                    ssim = tf.keras.metrics.Mean()
                    psnr = tf.keras.metrics.Mean()
                    for x_batch_test in test_dataset:
                        out, losses = model(x_batch_test)
                        ssim(tf.reduce_mean(tf.image.ssim(x_batch_test, out, 1.0)))
                        psnr(tf.reduce_mean(tf.image.psnr(x_batch_test, out, 1.0)))
                        test_loss(losses["loss"])
                    ssim_val = ssim.result()
                    psnr_val = psnr.result()
                    test_loss_mean = test_loss.result()
                    with test_summary_writer.as_default():
                        tf.summary.scalar('test_loss', test_loss_mean, step=iters // train_args.test_freq)
                        tf.summary.scalar('ssim', ssim_val, step=iters // train_args.test_freq)
                        tf.summary.scalar('psnr', psnr_val, step=iters // train_args.test_freq)

                    print('Epoch: {}, Test set loss: {}'.format(epoch, test_loss_mean))

            if iters % train_args.visualise_freq == 0:
                if test_dataset is not None:
                    generate_images(model, random_sample, train_args.results_dir, 'generated')
                    test_images(model, test_sample, train_args.results_dir)

            if iters % train_args.save_model_freq == 0:
                print("Saving model...")
                model.save_weights(train_args.model_dir)
                print("Model saved.")

            iters += 1





def arg_parser():

    with open('config/train_config.yml', 'r') as file:
        train_args = yaml.safe_load(file)

    with open('config/model_config.yml', 'r') as file:
        model_args = yaml.safe_load(file)

    parser_train = argparse.ArgumentParser()
    parser_model = argparse.ArgumentParser()

    parser_train.add_argument('--train_size', type=int, default=train_args['train_size'], help="Size of training set")
    parser_train.add_argument('--test_size', type=int, default=train_args['test_size'], help="Size of test set.")
    parser_train.add_argument('--batch_size', type=int, default=train_args['batch_size'], help="Training batch size.")
    parser_train.add_argument('--min_lr', type=float, default=train_args['min_lr'], help="Minimum learning rate.")
    parser_train.add_argument('--max_lr', type=float, default=train_args['max_lr'], help="Maximum learning rate.")
    parser_train.add_argument('--lr_step_size', type=int, default=train_args['lr_step_size'], help="Learning rate step size.")
    parser_train.add_argument('--epochs', type=int, default=train_args['epochs'], help="Number of epochs.")
    parser_train.add_argument('--load_model', type=bool, default=train_args['load_model'], help="Load existing model.")
    parser_train.add_argument('--num_samples', type=int, default=train_args['num_samples'], help="Number of random samples to generate.")
    parser_train.add_argument('--test_freq', type=int, default=train_args['test_freq'], help="Test every test_freq batch iters.")
    parser_train.add_argument('--save_model_freq', type=int, default=train_args['save_model_freq'], help="Save model every save_model_freq batch iters.")
    parser_train.add_argument('--visualise_freq', type=int, default=train_args['visualise_freq'], help="Visualise performance every # batch iters.")
    parser_train.add_argument('--log_freq', type=int, default=train_args['log_freq'], help="Log to tensorboard while training every # batch iters.")
    parser_train.add_argument('--model_dir', type=str, default=train_args['model_dir'], help="Saved model location.")
    parser_train.add_argument('--logs_dir', type=str, default=train_args['logs_dir'], help="Logs directory location.")
    parser_train.add_argument('--results_dir', type=str, default=train_args['results_dir'], help="Results directory location.")
    parser_train.add_argument('--tag', type=str, default=train_args['tag'], help="A tag for the method.")
    parser_train.add_argument('--dataset', type=str, default=train_args['dataset'], help="Dataset name.")

    parser_model.add_argument('--latent_dim', type=int, default=model_args['latent_dim'], help="Size of latent dimension")
    parser_model.add_argument('--amp_coef', type=float, default=model_args['amp_coef'], help="Amplitude component coefficient")
    parser_model.add_argument('--arg_coef', type=float, default=model_args['arg_coef'], help="Argument component coefficient")
    parser_model.add_argument('--ssim_coef', type=float, default=model_args['ssim_coef'], help="SSIM component coefficient")
    parser_model.add_argument('--beta', type=float, default=model_args['beta'], help="KL divergence coefficient")
    parser_model.add_argument('--stft_coef', type=float, default=model_args['stft_coef'], help="STFT loss coefficient")
    parser_model.add_argument('--eps', type=float, default=model_args['eps'], help="Epsilon value for stability")
    parser_model.add_argument('--stft_f', type=int, default=model_args['stft_f'], help="STFT window size")
    parser_model.add_argument('--stft_s', type=int, default=model_args['stft_s'], help="STFT stride length")
    parser_model.add_argument('--inp_dim', type=int, default=None, help="Input image dimension")
    parser_model.add_argument('--inp_c', type=int, default=None, help="Number of channels in input image")
    parser_model.add_argument('--dataset', type=str, default=None, help="Dataset name")
    parser_model.add_argument('--loss', type=str, default=model_args['loss'], help="Loss function")

    return parser_train, parser_model



if __name__ == '__main__':
    main()


