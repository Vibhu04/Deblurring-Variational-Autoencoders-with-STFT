import tensorflow as tf
from vae.vae_base import BaseVAE


class VAE2(BaseVAE):

    def __init__(self, args=None):

        super(VAE2, self).__init__()

        self.latent_dim = 200
        self.inp_shape = (256, 256, 3)
        self.encoder = self.encoder_(args)
        self.decoder = self.decoder_(args)

        super(VAE2, self).__init__(self.encoder, self.decoder, args)


    def encoder_(self, args):

        inputs = tf.keras.Input(shape=(256,256,3), name='input_layer')

        # Block-1
        x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', name='conv_1')(inputs)
        x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_1')(x)

        # Block-2
        x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_2')(x)

        # Block-3
        x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_3')(x)

        # Block-4
        x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', name='conv_4')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_4')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_4')(x)

        # Block-5
        x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', name='conv_5')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_5')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_5')(x)

        # Final Block
        flatten = tf.keras.layers.Flatten()(x)
        mean = tf.keras.layers.Dense(self.latent_dim, name='mean')(flatten)
        log_var = tf.keras.layers.Dense(self.latent_dim, name='log_var')(flatten)
        encoder = tf.keras.Model(inputs, (mean, log_var), name="Encoder")

        return encoder


    def decoder_(self, args):

        inputs = tf.keras.Input(shape=(self.latent_dim,), name='input_layer')
        x = tf.keras.layers.Dense(4096, name='dense_1')(inputs)
        x = tf.keras.layers.Reshape((8, 8, 64), name='Reshape')(x)

        # Block-1
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='conv_transpose_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_1')(x)

        # Block-2
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='conv_transpose_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_2')(x)

        # Block-3
        x = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same', name='conv_transpose_3')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_3')(x)

        # Block-4
        x = tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_4')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_4')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_4')(x)

        # Block-5
        outputs = tf.keras.layers.Conv2DTranspose(3, 3, 2, padding='same', activation='sigmoid', name='conv_transpose_5')(x)
        decoder = tf.keras.Model(inputs, outputs, name="Decoder")

        return decoder

