import tensorflow as tf
import numpy as np

class BaseVAE(tf.keras.Model):

    def __init__(self, encoder=None, decoder=None, args=None):

        super(BaseVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        if args is not None:
            self.dataset = args.dataset
            self.amp_coef = args.amp_coef
            self.arg_coef = args.arg_coef
            self.ssim_coef = args.ssim_coef
            self.beta = args.beta
            self.stft_coef = args.stft_coef
            self.eps = args.eps
            self.stft_f = args.stft_f
            self.stft_s = args.stft_s
            self.loss = args.loss
            self.inp_dim = args.inp_dim
            self.use_hann = False
            self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            self.hann_2d = self.get_hann_window()
            self.freq_filter = self.get_freq_filter()



    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)


    def encode(self, x):
        encoded = self.encoder(x)
        if self.dataset == 'mnist':
            mean = encoded[:, :self.latent_dim]
            logvar = encoded[:, self.latent_dim:]
        elif self.dataset == 'cartoons':
            mean = encoded[0]
            logvar = encoded[1]

        return mean, logvar


    def reparameterise(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        z = mean + tf.math.exp(logvar * 0.5) * eps

        return z


    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


    def get_hann_window(self):

        hann = tf.cast(tf.expand_dims(np.hanning(self.stft_f), axis=0), tf.float32)
        hann_2d = tf.expand_dims(tf.sqrt(tf.transpose(hann) * hann), axis=2)

        return hann_2d


    def get_freq_filter(self):

        lin = tf.expand_dims(tf.linspace(0.0, 1.0, int(self.stft_f/2)), axis=1)
        freq_filter = tf.expand_dims(lin * tf.transpose(lin), axis=2)

        return freq_filter


    @tf.function
    def stft(self, image, f=12, s=4):

        n = image.shape[1]
        num_conv = int((n - f) / s + 1)
        results = []
        half_f = int(f/2)
        for i in range(num_conv):
            for j in range(num_conv):
                img_segment = image[:, i * s:i * s + f, j * s:j * s + f, :]
                if self.use_hann:
                    fft = tf.signal.fft2d(tf.cast(img_segment * self.hann_2d, tf.complex64))
                else:
                    fft = tf.signal.fft2d(tf.cast(img_segment, tf.complex64))
                results.append(fft[:, :half_f, :half_f])
        results = tf.stack(results)
        results = tf.transpose(results, [1, 0, 2, 3, 4])

        return results


    @tf.function
    def calc_loss(self, mean, logvar, x_in, x_out):

        var = tf.math.exp(logvar)
        kld_loss = -0.5 * tf.reduce_sum(1 + tf.math.log(var) - var - tf.math.square(mean), axis=1)

        if self.loss == 'l2' or self.loss == 'l1' or self.loss == 'ssim':
            if self.loss == 'l2':
                recons_loss = tf.reduce_mean(tf.math.squared_difference(x_out, x_in), axis=[1,2,3])
            elif self.loss == 'l1':
                recons_loss = tf.reduce_mean(tf.abs(x_out - x_in), axis=[1, 2, 3])
            elif self.loss == 'ssim':
                recons_loss = self.ssim_coef * tf.image.ssim(x_in, x_out, 1.0)
            loss = tf.math.reduce_mean(self.beta * kld_loss + recons_loss)

        elif self.loss == 'dft+ssim' or self.loss == 'ours':
            recons_loss = self.ssim_coef * tf.image.ssim(x_in, x_out, 1.0)
            if self.loss == 'dft+ssim':
                stft_out = self.stft(x_out, self.inp_dim, 0)
                stft_in = self.stft(x_in, self.inp_dim, 0)
            elif self.loss == 'ours':
                stft_out = self.stft(x_out, self.stft_f, self.stft_s)
                stft_in = self.stft(x_in, self.stft_f, self.stft_s)
            stft_in_arg = tf.math.angle(stft_in + self.eps)
            stft_out_arg = tf.math.angle(stft_out + self.eps)
            stft_in_amp = tf.abs(stft_in)
            stft_out_amp = tf.abs(stft_out)
            stft_arg_loss = tf.reduce_sum(tf.abs(stft_out_arg - stft_in_arg) * self.freq_filter, axis=[1, 2, 3, 4])
            stft_amp_loss = tf.reduce_sum(tf.abs(stft_out_amp - stft_in_amp) * self.freq_filter, axis=[1, 2, 3, 4])
            stft_loss = self.arg_coef * stft_arg_loss + self.amp_coef * stft_amp_loss
            loss = tf.math.reduce_mean(self.beta * kld_loss + recons_loss + self.stft_coef * stft_loss)

        losses = {
            "loss" : loss,
            "kld_loss" : tf.math.reduce_mean(self.beta * kld_loss),
            "recons_loss" : tf.math.reduce_mean(recons_loss),
            "stft_loss" : tf.math.reduce_mean(self.stft_coef * stft_loss)
        }

        return losses



    @tf.function
    def call(self, x_in):

        tf.compat.v1.enable_eager_execution()

        mean, logvar = self.encode(x_in)
        z = self.reparameterise(mean, logvar)
        x_out = self.decode(z)

        losses = self.calc_loss(mean, logvar, x_in, x_out)

        return x_out, losses











