from keras.layers import Conv1D, Conv2D, UpSampling2D, InputLayer, MaxPooling2D, Input, RepeatVector, Reshape, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam


def alpha_v2():
    print("*****alpha_v2*****")
    model = Sequential()
    model.add(InputLayer(input_shape=(256, 256, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
    return model


def beta_v2():
    print("*****beta_v2*****")
    model = Sequential()
    model.add(InputLayer(input_shape=(256, 256, 1)))

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv1D(2, 3, activation='tanh', padding='same'))
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
    return model


def embed_vgg16():
    print("*****embed_vgg16*****")
    encoder_input = Input(shape=(256, 256, 1,))

    encoder_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
    encoder_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_1)
    encoder_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_1)

    encoder_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_1)
    encoder_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_2)
    encoder_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_2)

    encoder_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_2)
    encoder_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_3)
    encoder_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_3)
    encoder_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_3)

    encoder_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_3)
    encoder_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_4)
    encoder_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_4)

    embed_input = Input(shape=(1000,))
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_4, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

    decoder_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
    decoder_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_4)
    decoder_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_4)

    decoder_3 = UpSampling2D((2, 2))(decoder_4)
    decoder_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_3)
    decoder_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_3)
    decoder_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_3)

    decoder_2 = UpSampling2D((2, 2))(decoder_3)
    decoder_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_2)
    decoder_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_2)

    decoder_1 = UpSampling2D((2, 2))(decoder_2)
    decoder_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_1)
    decoder_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_1)

    decoder_output = Conv1D(2, 3, activation='tanh', padding='same')(decoder_1)
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='mse', metrics=['accuracy'])
    return model
