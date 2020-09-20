'''
  Variational Autoencoder (VAE) with the Keras Functional API.
'''

import keras
from keras.layers import Conv1D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, Layer, LeakyReLU, UpSampling1D, AveragePooling1D, Activation
from keras.layers import BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import pad

from graph import ECG_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

from utils import *
from config import get_config

from imblearn.over_sampling import SMOTE

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation='relu', name='conv12d'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, activation='relu', name = name)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

class ReflectionPadding1D(Layer):
    def __init__(self, padding=(64, 64), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return pad(input_tensor,  [[0, 0], [padding_left, padding_right], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super(ReflectionPadding1D, self).get_config()
        #print(config)
        return config

class ReflectionPadding1D_decode(Layer):
    def __init__(self, padding=(128, 128), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding1D_decode, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return pad(input_tensor,  [[0, 0], [padding_left, padding_right], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super(ReflectionPadding1D_decode, self).get_config()
        #print(config)
        return config
# Load dataset
config = get_config()

(X,y) = loaddata_nosplit_scaled_std(config.input_size, config.feature)
classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']#['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
from sklearn.model_selection import train_test_split
X, Xval, y, yval = train_test_split(X, y, test_size=0.25, random_state=1)

if config.smote:
    print(SMOTE)
    sm = SMOTE(sampling_strategy = 'auto', random_state=12)
    X, y = sm.fit_sample(X, y)

Xe = np.expand_dims(X, axis=2)
Xvale = np.expand_dims(Xval, axis=2)
#(m, n) = y.shape
#y = y.reshape((m, 1, n ))
#(mvl, nvl) = yval.shape
#yval = yval.reshape((mvl, 1, nvl))
import pandas as pd
y = np.array(pd.DataFrame(y).idxmax(axis=1))
yval = np.array(pd.DataFrame(yval).idxmax(axis=1))

target_train = y
target_test = yval 
# Data & model configuration
batch_size = config.batch
no_epochs = config.ae_epochs
verbosity = 1
latent_dim = 2
num_channels = 1


# Reshape data
input_train = Xe
input_test = Xvale
input_shape = (config.input_size, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')


# # =================
# # Encoder
# # =================

kernel_size =16
s = 2

def encoder_conv_block(inputs, config):
    kernel_size =16
    s = 2
    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=2,
        kernel_initializer='he_normal',
        activation=LeakyReLU(alpha=0.2))(inputs)
    layer = ReflectionPadding1D()(layer)

    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=2,
        kernel_initializer='he_normal')(layer)
    layer = Activation(LeakyReLU(alpha=0.2))(layer)
    layer = ReflectionPadding1D()(layer)

    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=2,
        kernel_initializer='he_normal')(layer)
    layer = Activation(LeakyReLU(alpha=0.2))(layer)
    layer = ReflectionPadding1D()(layer)

    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=2,
        kernel_initializer='he_normal')(layer)
    layer = Activation(LeakyReLU(alpha=0.2))(layer)
    layer = ReflectionPadding1D()(layer)

    return layer

def decoder_conv_block(inputs, config):

    #layer = Dense(config.input_size* 32, activation =LeakyReLU(alpha=0.2))(inputs)
    #layer = Reshape((config.input_size, 1))(layer)

    kernel_size = 8
    s = 2

    layer = UpSampling1D(size=2)(inputs)
    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=s,
        kernel_initializer='he_normal',
        )(layer)
    layer = ReflectionPadding1D_decode()(layer)
    layer = AveragePooling1D()(layer)
    layer = Activation(LeakyReLU(alpha=0.2))(layer)

    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=s,
        kernel_initializer='he_normal',
        )(layer)
    layer = ReflectionPadding1D_decode()(layer)
    layer = AveragePooling1D()(layer)
    layer = Activation(LeakyReLU(alpha=0.2))(layer)

    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=s,
        kernel_initializer='he_normal',
        )(layer)
    layer = ReflectionPadding1D_decode()(layer)
    layer = AveragePooling1D()(layer)
    layer = Activation(LeakyReLU(alpha=0.2))(layer)

    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=s,
        kernel_initializer='he_normal',
        )(layer)
    layer = ReflectionPadding1D_decode()(layer)
    layer = AveragePooling1D()(layer)
    layer = Activation(LeakyReLU(alpha=0.2))(layer)

    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=config.filter_length,
        kernel_size=kernel_size,
        padding='same',
        strides=s,
        kernel_initializer='he_normal',
        )(layer)

    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(1, LeakyReLU(alpha=0.2)))(layer)

    return layer

# Definition
i       = Input(shape=input_shape, name='encoder_input')
layer = encoder_conv_block(i, config)
x       = Flatten()(layer)
x       = Dense(20, activation='relu')(x)
x       = BatchNormalization()(x)
mu      = Dense(latent_dim, name='latent_mu')(x)
sigma   = Dense(latent_dim, name='latent_sigma')(x)

# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape = K.int_shape(layer)
print(conv_shape)
# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch     = K.shape(mu)[0]
  dim       = K.int_shape(mu)[1]
  eps       = K.random_normal(shape=(batch, dim))
  return mu + K.exp(sigma / 2) * eps

# Use reparameterization trick to ....??
z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

# Instantiate encoder
encoder = Model(i, [mu, sigma, z], name='encoder')
encoder.summary()

# =================
# Decoder
# =================

# Definition
d_i   = Input(shape=(latent_dim, ), name='decoder_input')
x     = Dense(conv_shape[1] * conv_shape[2], activation='relu')(d_i)
x     = BatchNormalization()(x)
x     = Reshape((conv_shape[1], conv_shape[2]))(x)
o     = decoder_conv_block(x, config)
# Instantiate decoder
decoder = Model(d_i, o, name='decoder')
decoder.summary()

# =================
# VAE as a whole
# =================

# Instantiate VAE
vae_outputs = decoder(encoder(i)[2])
vae         = Model(i, vae_outputs, name='vae')
vae.summary()

# Define loss
def kl_reconstruction_loss(true, pred):
  # Reconstruction loss
  reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * 256
  # KL divergence loss
  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  # Total loss = 50% rec + 50% KL divergence loss
  return K.mean(reconstruction_loss + kl_loss)

# Compile VAE
vae.compile(optimizer='adam', loss=kl_reconstruction_loss)

# Train autoencoder
vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_data = (input_test, input_test))
#vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)

# =================
# Results visualization
# Credits for original visualization code: https://keras.io/examples/variational_autoencoder_deconv/
# (François Chollet).
# Adapted to accomodate this VAE.
# =================
def viz_latent_space(encoder, data):
  input_data, target_data = data
  mu, _, _ = encoder.predict(input_data)
  plt.figure(figsize=(8, 10))
  scatter = plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
  plt.xlabel('z - dim 1')
  plt.ylabel('z - dim 2')
  plt.legend(handles = scatter.legend_elements()[0],labels=  classes)
  plt.show()

def viz_decoded(encoder, decoder, data):
  num_samples = 15
  figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
  grid_x = np.linspace(-4, 4, num_samples)
  grid_y = np.linspace(-4, 4, num_samples)[::-1]
  for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]])
          x_decoded = decoder.predict(z_sample)
          digit = x_decoded[0].reshape(img_width, img_height, num_channels)
          figure[i * img_width: (i + 1) * img_width,
                  j * img_height: (j + 1) * img_height] = digit
  plt.figure(figsize=(10, 10))
  start_range = img_width // 2
  end_range = num_samples * img_width + start_range + 1
  pixel_range = np.arange(start_range, end_range, img_width)
  sample_range_x = np.round(grid_x, 1)
  sample_range_y = np.round(grid_y, 1)
  plt.xticks(pixel_range, sample_range_x)
  plt.yticks(pixel_range, sample_range_y)
  plt.xlabel('z - dim 1')
  plt.ylabel('z - dim 2')
  # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
  # So reshape if necessary
  fig_shape = np.shape(figure)
  if fig_shape[2] == 1:
    figure = figure.reshape((fig_shape[0], fig_shape[1]))
  # Show image
  plt.imshow(figure)
  plt.show()

def plot_some_signals(vae, data):
    x_vae_pred = vae.predict(data)

    from matplotlib import pyplot as plt
    xaxis = np.arange(0,config.input_size)
    for count in range(5):
        plt.plot(xaxis, x_vae_pred[count])
    plt.title("vae reconstructed beats")
    plt.xlabel("beat length")
    plt.ylabel("signal")
    plt.show()

# Plot results
data = (input_test, target_test)
viz_latent_space(encoder, data)
plot_some_signals(vae, input_test)


Xde = vae.predict(input_train)
Xvalde = vae.predict(input_test)
target_train = np.array(pd.get_dummies(target_train))
target_test = np.array(pd.get_dummies(target_test))

(m, n) = target_train.shape
target_train = target_train.reshape((m, 1, n ))
(mvl, nvl) = target_test.shape
target_test = target_test.reshape((mvl, 1, nvl))

callbacks = [
        EarlyStopping(patience = config.patience, verbose=1),
        ReduceLROnPlateau(factor = 0.5, patience = 3, min_lr = 0.01, verbose=1),
        TensorBoard( log_dir='./logs', histogram_freq=0, write_graph = True, write_grads=False, write_images=True),
        ModelCheckpoint('models/{}-vae-2-latest.hdf5'.format(config.feature), monitor='val_loss', save_best_only=False, verbose=1, period=10)
]

initial_epoch = 0
model = ECG_model(config)
model.fit(Xde, target_train,
        validation_data=(Xvalde, target_test),
        epochs=config.epochs,
        batch_size=config.batch,
        callbacks=callbacks,
        initial_epoch=initial_epoch)
print_results(config, model, Xvalde, target_test, classes, "vae-2-")