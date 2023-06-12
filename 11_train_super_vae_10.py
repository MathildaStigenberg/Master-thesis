from collections import OrderedDict
import sklearn
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, normalize
from tensorflow import keras
from keras.models import Model
from keras import initializers
from keras import Input
from keras.layers import Dense, Lambda, Dropout, LeakyReLU, BatchNormalization
from keras import backend as K 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scanpy as sc
import pandas as pd
from statistics import mean, stdev, sqrt
from operator import add
import random

output = sys.argv[1]

input = sys.argv[2]

excluded_genes = sys.argv[3]

dat = sc.read_h5ad(input)

df = dat.to_df(layer='normalized')

# Exclude G2M related genes
genes_to_exclude = open(excluded_genes)
for gene in genes_to_exclude:
    for i in gene.split('\n'):
        if i in list(df.columns):
            df.drop(i, axis=1, inplace=True)

df.insert(loc = 0, column = 'g2m_per_spot', value = [i for i in dat.obs['g2m_per_spot']])

X = df.iloc[:,1:]
Y = df.iloc[:,0]

X = normalize(X, norm='l2')

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# shuffle X and Y
X = sklearn.utils.shuffle(X, random_state=42)
Y = sklearn.utils.shuffle(Y, random_state=42)

input_dim = X.shape[1]

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 10))
    return z_mean + K.exp(0.5*z_log_sigma) * epsilon

# Create model
def create_model(input_dim):
    # Encoder
    input_l = Input(shape=(input_dim,), name='encoder-input-layer')
    l = Dropout(0.8)(input_l)

    l = Dense(units=640, activation='elu', kernel_initializer='he_uniform')(l)
    l = BatchNormalization()(l)
    l = Dropout(0.2)(l)

    l = Dense(units=320, activation='elu', kernel_initializer='he_uniform')(l)
    l = BatchNormalization()(l)
    l = Dropout(0.2)(l)

    l = Dense(units=160, activation='elu', kernel_initializer='he_uniform')(l)
    l = BatchNormalization()(l)
    l = Dropout(0.2)(l)

    #Latent space
    z_mean = Dense(units=10, name='z-mean')(l) # Mean component
    z_log_sigma = Dense(units=10, name='z-log-sigma')(l) # Standard deviation component
    z = Lambda(sampling, name='z-sampling-layer')([z_mean, z_log_sigma]) # Z sampling layer

    # Create encoder model
    encoder = Model(input_l, [z_mean, z_log_sigma, z], name='encoder')

    # Decoder
    latent_space = Input(shape=(10,), name='input-z-sampling')
    l = Dropout(0.9)(latent_space)

    l = Dense(units=160, activation='elu', kernel_initializer='he_uniform')(l)
    l = BatchNormalization()(l)
    l = Dropout(0.8)(l)

    l = Dense(units=320, activation='elu', kernel_initializer='he_uniform')(l)
    l = BatchNormalization()(l)
    l = Dropout(0.8)(l)

    l = Dense(units=640, activation='elu', kernel_initializer='he_uniform')(l)
    l = BatchNormalization()(l)
    l = Dropout(0.8)(l)

    out = Dense(input_dim, activation='sigmoid', name='decoder-output-layer')(l)

    # Create decoder model
    decoder = Model(latent_space, out, name='decoder')

    # Regression 
    reg_latent_space = Input(shape=(10,), name='input-z-sampling-regression')
    l = Dropout(0.8)(reg_latent_space)

    l = Dense(units=10, activation='elu', kernel_initializer='he_uniform')(l)
    l = BatchNormalization()(l)
    l = Dropout(0.1)(l)

    reg_output = Dense(units=1, name='regression-output')(l)

    # Create supervised model
    vae_reg = Model(reg_latent_space, reg_output, name='supervised')

    # Create variational autoencoder model
    reconstruction = [decoder(encoder(input_l)[2]), vae_reg(encoder(input_l)[2])]
    vae = Model(input_l, reconstruction, name='vae_supervised')
    
    # Reconstruction loss and KL divergence loss 
    r_loss = input_dim * keras.losses.mse(input_l, reconstruction[0])
    kl_loss =  -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)
    vae_loss = K.mean(r_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss={'supervised': 'mse'})
    return vae, encoder, decoder, vae_reg

vae, encoder, decoder, vae_reg = create_model(input_dim)

# Train model
monitor = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = vae.fit(X, {'decoder': X, 'supervised': Y}, epochs = 1000, batch_size = 32, callbacks=[monitor])

# Plot loss charts
all_training = list(map(add, history.history['loss'], history.history['supervised_loss']))

fig, ax = plt.subplots()

plt.plot(all_training, label='Training loss', color='blue')
plt.title(label='Loss by epoch')
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.legend()
plt.savefig(output+'all_training_10.png')

fig, ax = plt.subplots()

plt.plot(history.history['loss'], label=' VAE training loss', color='blue')
plt.title(label='Loss by epoch')
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.legend()
plt.savefig(output+'vae_training_10.png')

fig, ax = plt.subplots()

plt.plot(history.history['supervised_loss'], label='Supervised training loss', color='blue')
plt.title(label='Loss by epoch')
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.legend()
plt.savefig(output+'supervised_training_10.png')

evaluation = vae.evaluate(X, Y)

fig, ax = plt.subplots()

reconstruct = vae.predict(X)
plt.scatter(Y,reconstruct[1], alpha=0.5, s=8)
plt.plot([0,80],[0,80], c='orange', linestyle='--')
plt.title('regression performance')
plt.ylabel('predicted values')
plt.xlabel('original values')
plt.grid(True)
plt.savefig(output+'supervised_regression_10.png')

fig, ax = plt.subplots()

spot = random.sample(range(X.shape[0]), 1)
reconstruct = vae.predict(X)
plt.scatter(X[spot],reconstruct[0][spot], alpha=0.5, s=8)
plt.plot([0,1],[0,1], c='orange', linestyle='--')
plt.title('VAE performance')
plt.ylabel('reconstructed values')
plt.xlabel('original values')
plt.grid(True)
plt.savefig(output+'vae_reconstruction_10.png')

super_r2 = r2_score(Y, vae.predict(X)[1])
vae_r2 = r2_score(X, vae.predict(X)[0])

rmse_score = sqrt(mean_squared_error(Y, vae.predict(X)[1]))
rmse_score_vae = sqrt(mean_squared_error(X, vae.predict(X)[0]))

with open(output+'performance_values_10.txt', 'w') as f:
    f.write('VAE evaluation loss: ')
    f.write(str(evaluation[0]))
    f.write('\n')
    f.write('Supervised evaluation loss: ')
    f.write(str(evaluation[1]))
    f.write('\n')
    f.write('Supervised RMSE: ')
    f.write(str(rmse_score))
    f.write('\n')
    f.write('VAE RMSE: ')
    f.write(str(rmse_score_vae))
    f.write('\n')
    f.write('Supervised r2 score: ')
    f.write(str(super_r2))
    f.write('\n')
    f.write('VAE r2 score: ')
    f.write(str(vae_r2))
    f.write('\n')

vae.save(output+'supervised_model_10.h5')
encoder.save(output+'supervised_encoder_10.h5')
decoder.save(output+'supervised_decoder_10.h5')
vae_reg.save(output+'supervised_reg_10.h5')
