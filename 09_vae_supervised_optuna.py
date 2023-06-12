from collections import OrderedDict
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import sklearn
import tensorflow as tf
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
from tensorflow import keras
from keras.models import Model, Sequential
from keras import initializers
from keras import Input
from keras.layers import Dense, Lambda, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import LecunUniform, LecunNormal
from keras import backend as K
import numpy as np
import sys
import os
import scanpy as sc
import pandas as pd
from statistics import mean, stdev
import plotly.express as px

spatial_data = sys.argv[1]
excluded_genes = sys.argv[2]
output = sys.argv[3]

dat = sc.read_h5ad(spatial_data)

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
Y = sklearn.utils.shuffle(Y, random_state=42)
X = sklearn.utils.shuffle(X, random_state=42)

input_dim = X.shape[1]

def sampling(args):
    z_mean, z_log_sigma, latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5*z_log_sigma) * epsilon

# Optuna optimizer
def create_model(trial):
    size = trial.suggest_categorical('size', [10240, 5120, 2560, 1280, 640])
    act_func = trial.suggest_categorical('activation', ['relu', 'LeakyReLU()', 'tanh', 'selu', 'elu'])
    n_hidden_layers = trial.suggest_int('HL_depth',1,5,step=1)
    n_hidden_layers_supervised = trial.suggest_int('HL_depth_s', 0,2,step=1)
    optimize = trial.suggest_categorical('optimizer', ['Adam()', 'SGD()'])
    latent_dim = trial.suggest_int('latent_dim',5,20,step=5)
    if act_func == 'relu' or act_func == 'LeakyReLU()' or act_func == 'elu':
        initializer = trial.suggest_categorical('initializer_he', ['he_normal', 'he_uniform'])
    elif act_func == 'selu':
        initializer = trial.suggest_categorical('initializer_lecun', ['LecunNormal()', 'LecunUniform()'])
    else:
        initializer = trial.suggest_categorical('initializer_glorot', ['glorot_normal', 'glorot_uniform'])

    #Encoder
    input_l = Input(shape=(input_dim,), name='encoder-input-layer')
    l = Dropout(rate=trial.suggest_categorical('DO_e', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(input_l)
    if act_func == 'LeakyReLU()':
        for i in range(0,n_hidden_layers):
            l = Dense(units=size/2**i, activation=eval(act_func), kernel_initializer=initializer)(l)
            l = BatchNormalization()(l)
            l = Dropout(rate=trial.suggest_categorical('DO_HL_e', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(l)
        #Latent space
        z_mean = Dense(units=latent_dim, name='z-mean')(l) # Mean component
        z_log_sigma = Dense(units=latent_dim, name='z-log-sigma')(l) # Standard deviation component
        z = Lambda(sampling, name='z-sampling-layer')([z_mean, z_log_sigma, latent_dim]) # Z sampling layer
        #Decoder
        latent_space = Input(shape=(latent_dim,), name='input-z-sampling')
        l = Dropout(rate=trial.suggest_categorical('DO_d', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(latent_space)
        for i in range(n_hidden_layers-1,0-1,-1):
            l = Dense(units=size/2**i, activation=eval(act_func), kernel_initializer=initializer)(l)
            l = BatchNormalization()(l)
            l = Dropout(rate=trial.suggest_categorical('DO_HL_d', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(l)
    else:
        if initializer == 'LecunNormal()' or initializer == 'LecunUniform()':
            for i in range(0,n_hidden_layers):
                l = Dense(units=size/2**i, activation=act_func, kernel_initializer=eval(initializer))(l)
                l = BatchNormalization()(l)
                l = Dropout(rate=trial.suggest_categorical('DO_HL_e', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(l)

            #Latent space
            z_mean = Dense(units=latent_dim, name='z-mean')(l) # Mean component
            z_log_sigma = Dense(units=latent_dim, name='z-log-sigma')(l) # Standard deviation component
            z = Lambda(sampling, name='z-sampling-layer')([z_mean, z_log_sigma, latent_dim]) # Z sampling layer

            #Decoder
            latent_space = Input(shape=(latent_dim,), name='input-z-sampling')
            l = Dropout(rate=trial.suggest_categorical('DO_d', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(latent_space)
            for i in range(n_hidden_layers-1,0-1,-1):
                l = Dense(units=size/2**i, activation=act_func, kernel_initializer=eval(initializer))(l)
                l = BatchNormalization()(l)
                l = Dropout(rate=trial.suggest_categorical('DO_HL_d', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(l)
        else:
            for i in range(0,n_hidden_layers):
                l = Dense(units=size/2**i, activation=act_func, kernel_initializer=initializer)(l)
                l = BatchNormalization()(l)
                l = Dropout(rate=trial.suggest_categorical('DO_HL_e', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(l)

            #Latent space
            z_mean = Dense(units=latent_dim, name='z-mean')(l) # Mean component
            z_log_sigma = Dense(units=latent_dim, name='z-log-sigma')(l) # Standard deviation component
            z = Lambda(sampling, name='z-sampling-layer')([z_mean, z_log_sigma, latent_dim]) # Z sampling layer

            #Decoder
            latent_space = Input(shape=(latent_dim,), name='input-z-sampling')
            l = Dropout(rate=trial.suggest_categorical('DO_d', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(latent_space)
            for i in range(n_hidden_layers-1,0-1,-1):
                l = Dense(units=size/2**i, activation=act_func, kernel_initializer=initializer)(l)
                l = BatchNormalization()(l)
                l = Dropout(rate=trial.suggest_categorical('DO_HL_d', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(l)

    out = Dense(input_dim, activation=trial.suggest_categorical('out_activation', ['sigmoid', 'linear']), name='decoder-output-layer')(l)
    #Supervised part
    reg_latent_space = Input(shape=(latent_dim,), name='input-z-sampling-regression')
    sl = Dropout(rate=trial.suggest_categorical('DO_s', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(reg_latent_space)
    if act_func == 'LeakyReLU()':
        for i in range(0, n_hidden_layers_supervised):
            sl = Dense(units=latent_dim, activation=eval(act_func), kernel_initializer=initializer)(sl)
            sl = BatchNormalization()(sl)
            sl = Dropout(rate=trial.suggest_categorical('DO_HL_s', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(sl)
    else:
        if initializer == 'LecunNormal()' or initializer == 'LecunUniform()':
            for i in range(0, n_hidden_layers_supervised):
                sl = Dense(units=latent_dim, activation=act_func, kernel_initializer=eval(initializer))(sl)
                sl = BatchNormalization()(sl)
                sl = Dropout(rate=trial.suggest_categorical('DO_HL_s', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(sl)
        else:
            for i in range(0, n_hidden_layers_supervised):
                sl = Dense(units=latent_dim, activation=act_func, kernel_initializer=initializer)(sl)
                sl = BatchNormalization()(sl)
                sl = Dropout(rate=trial.suggest_categorical('DO_HL_s', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))(sl)
    reg_output = Dense(units=1, name='regression-output')(sl)
        # Create encoder model
    encoder = Model(input_l, [z_mean, z_log_sigma, z], name='encoder')
    # Create decoder model
    decoder = Model(latent_space, out, name='decoder')
    # Create supervised model
    vae_reg = Model(reg_latent_space, reg_output, name='supervised')

     # Create variational autoencoder model
    reconstruction = [decoder(encoder(input_l)[2]), vae_reg(encoder(input_l)[2])]
    vae = Model(input_l, reconstruction, name='vae')

    # Reconstruction loss and KL divergence loss
    r_loss = input_dim * keras.losses.mse(input_l, reconstruction[0])
    kl_loss =  -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)
    vae_loss = K.mean(r_loss + kl_loss)
    vae.add_loss(vae_loss)
    if optimize == 'Adam()':
        vae.compile(optimizer=Adam(learning_rate=trial.suggest_categorical('learn_rate',[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])), loss={'supervised': 'mse'})
    else:
        vae.compile(optimizer=SGD(learning_rate=trial.suggest_categorical('learn_rate',[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])), loss={'supervised': 'mse'})
    return vae, encoder, decoder

def objective(trial):
    r2 = []
    train_loss = []
    eval_loss = []
    for train, val in KFold(n_splits=5).split(X):
        x_train, x_val = X[train], X[val]
        y_train, y_val = Y[train], Y[val]
        vae, encoder, decoder = create_model(trial)
        for previous_trial in trial.study.trials:
            if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
                print(f"Duplicated trial: {trial.params}, return {previous_trial.values}")
                return None
        monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = vae.fit(x_train, {'decoder': x_train, 'supervised': y_train}, epochs = 1000, batch_size = 32, validation_data=(x_val,{'decoder': x_val, 'supervised': y_val}), callbacks=[monitor])
        loss = history.history['loss']
        evaluation = vae.evaluate(x_val, y_val)
        reconstruction = vae.predict(x_val)
        score = r2_score(x_val, reconstruction[0])
        train_loss.append(loss[-1])
        eval_loss.append(evaluation[0])
        r2.append(score)
        tf.keras.backend.clear_session()
    return mean(eval_loss)

study = optuna.create_study(directions=["minimize"])
study.optimize(objective, n_trials=100, catch=(ValueError,))

trial = study.best_trials

with open(output + 'optuna_best_params_supervised_vae.txt','w') as f:
    if len(trial) == 1:
        f.write("- Value: ")
        f.write("\n")
        f.write("  Loss evaluation: " + str(trial[0].values[0]))
        f.write("\n")
        f.write("  Params: ")
        f.write("\n")
        for key, value in trial[0].params.items():
            f.write("    " + str(key) + ": " + str(value))
            f.write("\n")
    else:
        for i in range(0,len(trial)):
            f.write("- Values: ")
            f.write("\n")
            f.write("  Loss evaluation: " + str(trial[i].values[0]))
            f.write("\n")
            f.write("  Params: ")
            f.write("\n")
            for key, value in trial[i].params.items():
                f.write("    " + str(key) + ": " + str(value))
                f.write("\n")

params=["size", "HL_depth", "latent_dim", "activation", "DP_e", "DP_HL_e", "DP_HL_d", "DP_d", "out_activation", "learn_rate", "initializer_he", "initializer_lecun", "initializer_glorot", "optimizer", "HL_depth_s", "DO_s", "DO_HL_s"]

trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
all_params = {p_name for t in trials for p_name in t.params.keys()}

my_dic = {'eval_loss': [], 'initializer': []}
for i in all_params:
    if i != 'initializer_he' and i !='initializer_lecun' and i !='initializer_glorot':
        my_dic[i] = []

for i in trials:
    my_dic['eval_loss'] += [i.value]
    for k, v in my_dic.items():
        if k != 'eval_loss' and k != 'initializer':
            if k == 'DO_HL_s':
                if 'DO_HL_s' not in i.params:
                    my_dic[k] += [str('None')]
                else:
                    my_dic[k] += [str(i.params[k])]
            else:
                my_dic[k] += [str(i.params[k])]
    for p in all_params:
        if p == 'initializer_he' or p =='initializer_glorot' or p == 'initializer_lecun':
            if p in i.params:
                my_dic['initializer'] += [str(i.params[p])]

df_params = pd.DataFrame(data=my_dic)

df_params = df_params.sort_values(by='eval_loss', axis=0, ascending=False, ignore_index=True)

df_params.to_pickle(output + 'super_vae_params.pkl')
