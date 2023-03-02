import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import os
import datetime
import math



def create_writers(args):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = args.logs_dir + current_time + '/train'
    test_log_dir = args.logs_dir + current_time + '/test'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    test_writer = tf.summary.create_file_writer(test_log_dir)

    return train_writer, test_writer


def verify_dirs(args):

    args.model_dir += args.tag + '/' + args.tag
    args.logs_dir += args.tag + '/gradient_tape/'
    args.results_dir += args.tag

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)



def show_random(images):

    sample = random.randint(0, images.shape[0])
    plt.imshow(images[sample])
    plt.show()


def preprocess_images(images, dataset):

    if dataset == 'mnist':
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        images = np.where(images > 0.5, 1.0, 0.0).astype('float32')

    elif dataset == 'cartoons':
        images.reshape((images.shape[0], 256, 256, 3))

    return images


def get_datasets(train_args, model_args):

    if train_args.dataset == 'mnist':
        dataset = tf.keras.datasets.mnist
        model_args.inp_dim = 28
        model_args.inp_c = 1
        model_args.dataset = 'mnist'
        (train_images, _), (test_images, _) = dataset.load_data()
        train_images = preprocess_images(train_images, train_args.dataset)
        test_images = preprocess_images(test_images, train_args.dataset)

        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                         .shuffle(train_args.train_size).batch(train_args.batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(train_args.test_size).batch(train_args.batch_size))

    elif train_args.dataset == 'cartoons':
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
          'datasets/cartoonset10k',
          image_size=(256, 256),
          batch_size=train_args.batch_size,
          label_mode=None)
        model_args.inp_dim = 256
        model_args.inp_c = 3
        model_args.dataset = 'cartoons'
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale= 1./255)
        train_dataset = dataset.map(lambda x: normalization_layer(x))
        test_dataset = None

    return train_dataset, test_dataset


def generate_images(model, random_sample, results_dir, name):

    num_samples = random_sample.shape[0]
    dim = math.ceil(num_samples ** 0.5)

    images = model.sample(random_sample)

    for i in range(images.shape[0]):
        plt.subplot(dim, dim, i+1)
        plt.imshow(images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    file_name = results_dir + '/' + name + '.png'
    plt.savefig(file_name)
    print("Generated samples saved at " + file_name)



def test_images(model, test_images, results_dir):

    for i in range(8):
        plt.subplot(4, 4, 2*i+1)
        plt.imshow(test_images[i][:][:], cmap='gray')
        out_image, _ = model(tf.expand_dims(test_images[i][:][:], axis=0))
        plt.subplot(4, 4, 2 * i + 2)
        plt.imshow(out_image[0], cmap='gray')
        plt.axis('off')

    plt.savefig(results_dir + '/test.png')