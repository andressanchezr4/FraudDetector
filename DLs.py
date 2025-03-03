# -*- coding: utf-8 -*-
"""
Created on 2024

@author: andres.sanchez
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

        self.encoded_1 = layers.Dense(input_dim, activation='relu')
        self.bn_1 = layers.BatchNormalization()
        
        self.encoded_2 = layers.Dense(128, activation='relu')
        self.bn_2 = layers.BatchNormalization()
        
        self.encoded_3 = layers.Dense(64, activation='relu')
        self.bn_3 = layers.BatchNormalization()
        
        self.latent_space = layers.Dense(32, activation = 'relu')
    
    def call(self, inputs):

        encoded_1 = self.encoded_1(inputs)
        encoded_1 = self.bn_1(encoded_1)
        
        encoded_2 = self.encoded_2(encoded_1)
        encoded_2 = self.bn_2(encoded_2)
        
        encoded_3 = self.encoded_3(encoded_2)
        encoded_3 = self.bn_3(encoded_3)
        
        latent_space = self.latent_space(encoded_3)
        
        return latent_space

class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        
        self.decoded_1 = layers.Dense(64, activation='relu')
        self.bn_1 = layers.BatchNormalization()
        
        self.decoded_2 = layers.Dense(128, activation='relu')
        self.bn_2 = layers.BatchNormalization()
        
        self.decoded_input_data = layers.Dense(input_dim, activation = 'sigmoid')
    
    def call(self, latent_space):
        
        decoded_1 = self.decoded_1(latent_space)
        decoded_1 = self.bn_1(decoded_1)
        
        decoded_2 = self.decoded_2(decoded_1)
        decoded_2 = self.bn_2(decoded_2)
        
        decoded_data = self.decoded_input_data(decoded_2)
        
        return decoded_data

class Autoencoder(tf.keras.Model):
    def __init__(self, input_data, **kwargs):
        super().__init__()
        
        self.input_dim = input_data.shape[1]
        
        self.data_loss_tracker = keras.metrics.Mean(name="data_loss")
        
        self.encoder = Encoder(self.input_dim)
        self.decoder = Decoder(self.input_dim)
        
    @property
    def metrics(self):
        return [
            self.data_loss_tracker,
        ]
    
    def call(self, inputs):
        latent_space = self.encoder(inputs)
        decoded_data = self.decoder(latent_space)
        return decoded_data, latent_space
    
    def train_step(self, data):
        x, y = data
        y = tf.cast(y, dtype=tf.float32)
        x = tf.cast(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            latent_space = self.encoder(x)
            decoded_data = self.decoder(latent_space)
            
            data_loss = tf.reduce_mean(tf.math.squared_difference(x, decoded_data), 1)
            
        grads = tape.gradient(data_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.data_loss_tracker.update_state(data_loss)
        
        return {
                "data_loss": self.data_loss_tracker.result(),
                }
    
    def test_step(self, val_data):
        x, y = val_data
        
        y = tf.cast(y, dtype=tf.float32)
        x = tf.cast(x, dtype=tf.float32)
        
        latent_space = self.encoder(x)
        decoded_data = self.decoder(latent_space)
        
        data_loss = tf.reduce_mean(tf.math.squared_difference(x, decoded_data))
        
        return {
                "data_loss": data_loss,
                }
    
    def plot_latent_space(self, x, y):
        
        _, latent_embeddings = self.predict(x)
        tsne_embeddings = TSNE(n_components=2).fit_transform(latent_embeddings)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=["red" if i else 'blue' for i in y.to_numpy().ravel()], alpha=0.7)
        plt.colorbar(scatter, label='Normalized Price')
        plt.title('Latent Space Colored by Price')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()
    
    def AE4AnomDetect(self, standardized_data_val, val_y):
        
        def mad_score(points):
            m = np.median(points)
            ad = np.abs(points - m)
            mad = np.median(ad)
            
            return 0.6745 * ad / mad
        
        decoded_data_val, latent_space_val = self.predict(standardized_data_val)
        mse = np.mean(np.power(standardized_data_val - decoded_data_val, 2), axis=1)
        
        val_y = val_y.reset_index(drop=True)
        clean = mse[val_y.isFraud==0]
        fraud = mse[val_y.isFraud==1]

        fig, ax = plt.subplots(figsize=(6,6))

        ax.hist(clean, bins=50, density=True, label="clean", alpha=.6, color="green")
        ax.hist(fraud, bins=50, density=True, label="fraud", alpha=.6, color="red")

        plt.title("(Normalized) Distribution of the Reconstruction Loss")
        plt.legend()
        plt.show()

        THRESHOLD = 1.96 
        
        z_scores = mad_score(mse)
        outliers = z_scores > THRESHOLD

        cm = confusion_matrix(val_y.isFraud.tolist(), outliers, labels=[True, False])
        
        print("Confusion Matrix:\n", cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[True, False])
        disp.plot(cmap="Blues")
        
        return cm

    
class LossThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("val_data_loss")
        if loss is not None and loss < self.threshold:
            print(f"\nStopping training as loss has reached {loss:.4f}, below the threshold of {self.threshold}")
            self.model.stop_training = True

def plot_training_loss(history):

        plt.plot(history['data_loss'], label='Train Loss')
        plt.plot(history['val_data_loss'], label='Validation Loss')
        plt.title('Loss over training')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=1000,  
    decay_rate=0.9, 
    staircase=True  
)
