import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

latent_dim = 12

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
    autoencoder_cifar.compile(optimizer='adam', loss='mse')
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