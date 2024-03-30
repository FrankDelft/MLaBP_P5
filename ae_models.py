import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import vcae_fmnist
import vcae_cifar
import torch
import os
from tqdm import tqdm
import numpy as np

latent_dim = 12
capacity=64
device = torch.device("cpu")
learning_rate = 1e-3

def get_AE_CIFAR_model():
    # Create the encoder
    encoder_input = keras.Input(shape=(32*32*3,))
    encoded1 = layers.Dense(580, activation='relu')(encoder_input)
    encoded2 = layers.Dense(256, activation='relu')(encoded1)
    encoded3 = layers.Dense(latent_dim, activation='relu')(encoded2)
    encoder = keras.Model(encoder_input, encoded3)
    encoder.summary()

    # Create the decoder
    decoder_input = keras.Input(shape=(latent_dim,))
    decoded1 = layers.Dense(256, activation='sigmoid')(decoder_input)
    decoded2 = layers.Dense(580, activation='sigmoid')(decoded1)
    decoded3 = layers.Dense(32*32*3, activation='sigmoid')(decoded2)
    decoder = keras.Model(decoder_input, decoded3)
    decoder.summary()

    # Create the autoencoder
    autoencoder_cifar = keras.Model(encoder_input, decoder(encoder(encoder_input)))
    autoencoder_cifar.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder_cifar.summary()

    return autoencoder_cifar

def get_AE_FMNIST_model():
    encoder_input = keras.Input(shape=(784,))
    encoded1 = layers.Dense(580, activation='relu')(encoder_input)
    encoded2 = layers.Dense(256, activation='relu')(encoded1)
    encoded3 = layers.Dense(latent_dim, activation='relu')(encoded2)
    encoder = keras.Model(encoder_input, encoded3)
    encoder.summary()

    # Create the decoder
    decoder_input = keras.Input(shape=(latent_dim,))
    decoded1 = layers.Dense(256, activation='sigmoid')(decoder_input)
    decoded2 = layers.Dense(580, activation='sigmoid')(decoded1)
    decoded3 = layers.Dense(784, activation='sigmoid')(decoded2)
    decoder = keras.Model(decoder_input, decoded3)
    decoder.summary()

    # Create the autoencoder
    autoencoder_fmnist = keras.Model(encoder_input, decoder(encoder(encoder_input)))
    autoencoder_fmnist.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder_fmnist.summary()

    return autoencoder_fmnist

def get_CAE_FMNIST_Model():
    input_img = keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(12, activation='relu')(x)

    x = layers.Dense(128, activation='relu')(encoded)
    x = layers.Reshape((4, 4, 8))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def get_CAE_CIFAR_Model():
    input_img = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(12, activation='relu')(x)

    x = layers.Dense(128, activation='relu')(encoded)
    x = layers.Reshape((4, 4, 8))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu',padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def get_VCAE_CIFAR_model():
    vae = vcae_cifar.VariationalAutoencoder(hidden_channels=capacity, latent_dim=latent_dim)
    vae = vae.to(device)
    return  vae

def train_model(model, model_name, x_train, validation):
    checkpoint_path = f"./training_checkpoints/{model_name}/"
    checkpoint_path += "{epoch:03d}-{val_loss:.4f}.keras"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=False, # Save entire model
                                verbose=1,
                                save_best_only=False,
                                save_freq="epoch")

    csv_logger = CSVLogger(f'./losses/{model_name}_losses.csv', append=True, separator=',')

    model.fit(x_train, x_train,
                    epochs=20,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(validation, validation),
                    callbacks=[csv_logger, cp_callback])
    return model

def train_torch_model(model, model_name, train_dataloader):

    num_epochs = 20
    checkpoint_path = f"./training_checkpoints/{model_name}/"
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5) 


    model.train()

    train_loss_avg = []
    mse_loss_avg = []


    tqdm_bar = tqdm(range(1, num_epochs+1), desc="epoch [loss: ...]")
    # tqdm_iter = trange(1, num_epochs+1, desc="epoch [loss: ...]")
    for epoch in tqdm_bar:
        train_loss_averager = vcae_cifar.make_averager()
        mse_loss_averager = vcae_cifar.make_averager()

        batch_bar =  tqdm(train_dataloader, leave=False, desc='batch', total=len(train_dataloader))
        for image_batch, _ in batch_bar:

            image_batch = image_batch.to(device)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = model(image_batch)

            # total loss and mse loss
            total_loss, mse_loss_val = vcae_cifar.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
    
            mse_loss_averager(mse_loss_val.item())  # Add the current MSE loss to the averager
            

            # backpropagation
            optimizer.zero_grad()
            total_loss.backward()

            # one step of the optmizer
            optimizer.step()

            vcae_cifar.refresh_bar(batch_bar, f"train batch [loss: {train_loss_averager(total_loss.item()):.3f}]")
            vcae_cifar.refresh_bar(batch_bar, f"epoch [mse_loss: {mse_loss_averager(None):.3f}]")  # Print the average MSE loss for this epoch
            

        vcae_cifar.refresh_bar(tqdm_bar, f"epoch [total loss: {train_loss_averager(None):.3f}, mse loss: {mse_loss_averager(None):.3f}]")
    

        train_loss_avg.append(train_loss_averager(None))
        mse_loss_avg.append(mse_loss_averager(None))

        # Save the model at the end of each epoch
        # Define the directory path
        directory = "training_checkpoints"


        # Save the model state and other parameters/metrics
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_loss': train_loss_averager(None),
            'mse_loss': mse_loss_averager(None),
        }, os.path.join(directory, f'vae_epoch_{epoch}_'+model_name+'.pth'))


    return model

def torch_predict(model,test_dataloader):
    model.eval()
    test_loss_averager = vcae_cifar.make_averager()
    images_recon = torch.Tensor().cpu()
    with torch.no_grad():
        test_bar = tqdm(test_dataloader, total=len(test_dataloader), desc = 'batch [loss: ...]')
        for image_batch, _ in test_bar:
            image_batch = image_batch.to(device)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = model(image_batch)
            images_recon = torch.cat((images_recon, image_batch_recon.cpu()), 0)
            # reconstruction error
            loss,mse_loss = vcae_cifar.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            vcae_cifar.refresh_bar(test_bar, f"test batch [loss: {test_loss_averager(loss.item()):.3f}]")
    return np.transpose(images_recon,(0,2,3,1))