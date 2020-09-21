'''
  Base for ae_multi-tri.
  Only has reconstruction loss.

'''

import keras
from keras.layers import Conv1D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, UpSampling1D, AveragePooling1D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy, mean_squared_error
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from config import get_config

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

config = get_config()

(X,y) = loaddata_nosplit_scaled(config.input_size, config.feature)
classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']#['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
Xe = np.expand_dims(X, axis=2)
from sklearn.model_selection import train_test_split
Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.25, random_state=1)

import pandas as pd
y = np.array(pd.DataFrame(y).idxmax(axis=1))
yval = np.array(pd.DataFrame(yval).idxmax(axis=1))

target_train = y
target_test = yval 

# Data & model configuration
batch_size = 256
no_epochs = 50
validation_split = 0.2
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

# Definition
i       = Input(shape=input_shape, name='encoder_input')
cx      = Conv1D(filters=8, kernel_size=16, strides=2, padding='same', activation='relu')(i)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=1, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
eo      = BatchNormalization()(cx)


conv_shape = K.int_shape(cx)
print(conv_shape)
# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch     = K.shape(mu)[0]
  dim       = K.int_shape(mu)[1]
  eps       = K.random_normal(shape=(batch, dim))
  return mu + K.exp(sigma / 2) * eps


# Instantiate encoder
encoder = Model(i, eo, name='encoder')
#encoder = Model(i, [mu, sigma, z], name='encoder')
encoder.summary()

# # =================
# # Encoder_2
# # =================

# Definition
i_2       = Input(shape=input_shape, name='encoder2_input')
cx      = Conv1D(filters=8, kernel_size=16, strides=2, padding='same', activation='relu')(i_2)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=1, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
eo_2      = BatchNormalization()(cx)

conv_shape_2 = K.int_shape(cx)

# Instantiate encoder
encoder_2 = Model(i_2, eo_2, name='encoder_2')
encoder_2.summary()

# =================
# Decoder
# =================

# Definition
d_i   = Input(shape=(conv_shape[1], conv_shape[2]*2), name='decoder_input')
cx    = UpSampling1D(size=2)(d_i)
cx    = Conv1D(filters=2, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=2)(cx)
cx    = Conv1D(filters=2, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=2)(cx)
cx    = Conv1D(filters=1, kernel_size=16, strides=2, padding='same',  activation='relu', name = 'conv12d3')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=4)(cx)
cx    = Conv1D(filters=1, kernel_size=16, strides=2, padding='same',  activation='relu', name = 'conv12d4')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=4)(cx)
cx    = Conv1D(filters=num_channels, kernel_size=16, activation='relu', padding='same', name='decoder_output')(cx)
o     = UpSampling1D(size=2)(cx)
# Instantiate decoder
decoder = Model(d_i, o, name='decoder')
decoder.summary()

# =================
# VAE as a whole
# =================
from tensorflow import concat
# Instantiate VAE
vae_outputs = decoder(concat([encoder(i), encoder_2(i)], 2))
vae         = Model(i, vae_outputs, name='multi-ae')
vae.summary()

# Define loss
def reconstruction_loss(true, pred):
  # Reconstruction loss
  reconstruction_loss = mean_squared_error(K.flatten(true), K.flatten(pred))# * 256
  return reconstruction_loss
  #return K.mean(reconstruction_loss + kl_loss)

# Compile VAE
vae.compile(optimizer='adam', loss=reconstruction_loss)

# Train autoencoder
vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_data = (input_test, input_test))
#vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)


# =================
# Results visualization
# Credits for original visualization code: https://keras.io/examples/variational_autoencoder_deconv/
# (François Chollet).
# =================
def viz_latent_space(encoder, data):
  input_data, target_data = data

  print(target_data.shape)
  #print(target_data.shape[0])
  #print(target_data)
  print(encoder.predict(input_data).shape)#.reshape((32,16))
  print(encoder.predict(input_data).reshape(input_data.shape[0], 16).shape)#.reshape((32,16))

  from sklearn.manifold import TSNE
  X_tsne = TSNE(n_components=2, random_state=1).fit_transform(encoder.predict(input_data).reshape(input_data.shape[0], 16))
  print(X_tsne.shape)
  print(X_tsne)


  plt.figure(figsize=(8, 10))
  scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=target_data, label = classes)
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)
  plt.title("tsne")
  plt.show()

def viz_latent_space_pca(encoder, data):
  input_data, target_data = data

  print(target_data.shape)
  #print(target_data.shape[0])
  #print(target_data)
  print(encoder.predict(input_data).shape)#.reshape((32,16))
  print(encoder.predict(input_data).reshape(input_data.shape[0], 16).shape)#.reshape((32,16))

  from sklearn.decomposition import PCA
  principalComponents = PCA(n_components=2, random_state = 1).fit_transform(encoder.predict(input_data).reshape(input_data.shape[0], 16)) 
  print(principalComponents.shape)
  print(principalComponents)   


  plt.figure(figsize=(8, 10))
  scatter = plt.scatter(principalComponents[:,0], principalComponents[:,1], c=target_data, label=classes)
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)
  plt.title("pca")
  plt.show()

def plot_some_signals(vae, data):
    x_vae_pred = vae.predict(data)

    from matplotlib import pyplot as plt
    xaxis = np.arange(0,config.input_size)
    for count in range(5):
        plt.plot(xaxis, x_vae_pred[count])
    plt.title("ae reconstructed beats")
    plt.xlabel("beat length")
    plt.ylabel("signal")
    plt.show()

# Plot results
data = (input_test, target_test)
viz_latent_space(encoder, data)
viz_latent_space(encoder_2, data)
viz_latent_space_pca(encoder, data)
viz_latent_space_pca(encoder2, data)

plot_some_signals()
#viz_decoded(encoder, decoder, data)