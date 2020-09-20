'''
  Variational Autoencoder (VAE) with the Keras Functional API.
'''

import keras
from keras.layers import Conv1D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, UpSampling1D, AveragePooling1D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy, mean_squared_error, CategoricalCrossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import deepdish.io as ddio


from utils import *
from config import get_config


config = get_config()

classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']


input_shape = (config.input_size, 1)

input_train = ddio.load('dataset/traindata_tri.hdf5')
target_train = ddio.load('dataset/trainlabel_tri.hdf5')
input_test = ddio.load('dataset/testdata_tri.hdf5')
target_test = ddio.load('dataset/testlabel_tri.hdf5')

# Data & model configuration
batch_size = 256
no_epochs = 10
validation_split = 0.25
verbosity = 1
latent_dim = 2
num_channels = 1


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
#cx      = BatchNormalization()(cx)
#cx      = Conv1D(filters=1, kernel_size=8, strides=2, padding='same', activation='relu')(cx)
eo      = BatchNormalization()(cx)


# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape = K.int_shape(cx)
print(conv_shape)

# Use reparameterization trick to ....??
#z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

# Instantiate encoder
encoder = Model(i, eo, name='encoder_pattern')
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
#cx      = BatchNormalization()(cx)
#cx      = Conv1D(filters=1, kernel_size=8, strides=2, padding='same', activation='relu')(cx)
eo_2      = BatchNormalization()(cx)


# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape_2 = K.int_shape(cx)

# Instantiate encoder
encoder_2 = Model(i_2, eo_2, name='encoder_subject')
encoder_2.summary()

# =================
# Decoder
# =================

# Definition
#d_i   = Input(shape=(latent_dim, ), name='decoder_input')
#d_i   = Input(shape=(conv_shape[1]*2, conv_shape[2]), name='decoder_input')
d_i   = Input(shape=(conv_shape[1], conv_shape[2]*2), name='decoder_input')
#x     = Dense(conv_shape[1] * conv_shape[2], activation='relu')(d_i)
#x     = BatchNormalization()(x)
#x     = Reshape((conv_shape[1], conv_shape[2]))(x)
#x     = Reshape((conv_shape[1], conv_shape[2]))(d_i)
cx    = UpSampling1D(size=2)(d_i)
#cx    = UpSampling1D(size=2)(cx)
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
import tensorflow as tf

class AE(keras.Model):
    def __init__(self, encoder, encoder_2, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_2 = encoder_2
        self.decoder = decoder
    
    def call(self, inputs):
        return self.decoder(tf.concat([self.encoder(inputs), self.encoder_2(inputs)], 2))

    def train_step(self, data):
        data = data[0]
        data_1 = data[1]
        true = data[2]
        data_p = data[3]
        data_s = data[4]
        data_r = data[5]

        with tf.GradientTape() as tape:
            encoder_output = encoder(data)
            encoder_2_output = encoder_2(data_1)
            reconstruction = decoder(tf.concat([encoder_output, encoder_2_output], 2))
            
            cross_recon_loss = tf.reduce_mean(
                mean_squared_error(true, reconstruction)
            )
            alpha = 0.2
            trip_p_loss = tf.reduce_mean(abs(encoder(data)-encoder(data_p))- abs(encoder(data)-encoder(data_r)) + alpha)
            trip_s_loss = tf.reduce_mean(abs(encoder_2(data_1)-encoder_2(data_s))- abs(encoder_2(data_1)-encoder_2(data_r)) + alpha)
            total_loss = cross_recon_loss + trip_s_loss + trip_p_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "cross_recon_loss": cross_recon_loss,
            "trip_s_loss": trip_s_loss,
            "trip_p_loss": trip_p_loss
        }


# Instantiate AE
#vae_outputs = decoder(tf.concat([encoder(i), encoder_2(i)], 2))
#vae         = AE(encoder, decoder, name='multi-ae')
ae         = AE(encoder, encoder_2, decoder, name='multi-ae')

# Compile VAE
ae.compile(optimizer=keras.optimizers.Adam())

# Train autoencoder
#vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_data = (input_test, input_test))
ae.fit([input_train[0], input_train[1], input_train[2], input_train[3], input_train[4], input_train[5]], epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)
ae.summary()



# =================
# Results visualization
# Credits for original visualization code: https://keras.io/examples/variational_autoencoder_deconv/
# (François Chollet).
# Adapted to accomodate this VAE.
# =================
def viz_latent_space(encoder, data, title):
  input_data, target_data = data
  print("tsne plot")

  from sklearn.manifold import TSNE
  X_tsne = TSNE(n_components=2, random_state=1).fit_transform(encoder.predict(input_data).reshape(input_data.shape[0], 16))

  plt.figure(figsize=(8, 10))
  scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=target_data, label = classes)
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)
  plt.title("tsne")
  plt.show()
  plt.savefig('results/ae_multi-tri-'+str(title)+'tsne.png')

def viz_latent_space_pca(encoder, data, title):
  input_data, target_data = data
  target_data = (np.array(pd.DataFrame(target_data.reshape(target_data.shape[0], target_data.shape[2])).idxmax(axis=1)))
  print('pca plot')

  from sklearn.decomposition import PCA
  principalComponents = PCA(n_components=2, random_state = 1).fit_transform(encoder.predict(input_data).reshape(input_data.shape[0], 16))
  
  plt.figure(figsize=(8, 10))
  '''
  for um in unique_markers:
      mask = m == um 
      um = "$" + um + '$'
      # mask is now an array of booleans that can be used for indexing  
      scatter = plt.scatter((principalComponents[:,0])[mask], (principalComponents[:,1])[mask], marker=um, c=target_data[:,0][mask], label=classes)
  '''
  scatter = plt.scatter((principalComponents[:,0]), (principalComponents[:,1]), c=target_data, label=classes)
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)
  plt.title(title)
  plt.show()
  plt.savefig('results/ae_multi-tri-'+str(title)+'pca.png')


def plot_some_signals(ae, data):
    input_data, target_data = data
    x_vae_pred = ae.predict(input_data)

    from matplotlib import pyplot as plt
    xaxis = np.arange(0,config.input_size)
    for count in range(5):
        plt.plot(xaxis, x_vae_pred[count])
    plt.title("ae reconstructed beats")
    plt.xlabel("beat length")
    plt.ylabel("signal")
    plt.show()
    plt.savefig('results/ae_multi-tri-vae-recon.png')


# Plot results
data = (input_test, target_test)
viz_latent_space(encoder, data, "tsne: pattern encoder")
viz_latent_space(encoder_2, data, "tsne: subject encoder")
viz_latent_space_pca(encoder, data, "pca: pattern encoder")
viz_latent_space_pca(encoder_2, data, "pca: subject encoder")
plot_some_signals(ae, data)

# Classifer Definition
i_c     = Input(shape=(conv_shape[1],conv_shape[2]), name='encoder2_input')
cx      = Reshape((conv_shape[1],))(i_c)
co      = Dense(len(classes), activation='softmax')(cx)

# Instantiate Classifer
classifier = Model(i_c, co, name='classifier')
classifier.summary()
classifier.compile(optimizer=keras.optimizers.Adam(), loss = tf.keras.losses.CategoricalCrossentropy())
classifier.fit(encoder(input_train[0]), target_train[0], epochs = no_epochs, validation_data = (encoder(input_test), target_test.reshape(target_test.shape[0],target_test.shape[1]*target_test.shape[2])))
#classifier.fit(encoder(input_train[0]), target_train[0].reshape(target_train[0].shape[0], 1, target_train[0].shape[1]), epochs = no_epochs, validation_data = (encoder(input_test), target_test))
print_results_ae_multi(config, classifier, encoder(input_test), target_test.reshape(target_test.shape[0],target_test.shape[1]*target_test.shape[2]), classes)
