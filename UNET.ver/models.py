from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, RepeatVector, Conv2DTranspose, concatenate, LeakyReLU
from keras.models import Model
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam


def unet_vgg16():
    encoder_input = Input(shape=(256, 256, 1,))

    encoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
    encoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_c1)
    encoder_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c1)

    encoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_p1)
    encoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_c2)
    encoder_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c2)

    encoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_p2)
    encoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_c3)
    encoder_p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c3)

    encoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_p3)
    encoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_c4)
    encoder_p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c4)

    middle_in = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_p4)
    middle_in = Conv2D(256, (3, 3), activation='relu', padding='same')(middle_in)
    embed_input = Input(shape=(1000,))
    fusion_output = RepeatVector(16 * 16)(embed_input)
    fusion_output = Reshape(([16, 16, 1000]))(fusion_output)
    fusion_output = concatenate([middle_in, fusion_output], axis=3)
    fusion_output = Conv2D(512, (1, 1), activation='relu', padding='same')(fusion_output)
    middle_out = Conv2D(256, (3, 3), activation='relu', padding='same')(fusion_output)

    decoder_c4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(middle_out)
    decoder_c4 = concatenate([decoder_c4, encoder_c4])
    decoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_c4)
    decoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_c4)

    decoder_c3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(decoder_c4)
    decoder_c3 = concatenate([decoder_c3, encoder_c3])
    decoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_c3)
    decoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_c3)

    decoder_c2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(decoder_c3)
    decoder_c2 = concatenate([decoder_c2, encoder_c2])
    decoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_c2)
    decoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_c2)

    decoder_c1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(decoder_c2)
    decoder_c1 = concatenate([decoder_c1, encoder_c1])
    decoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_c1)
    decoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_c1)

    decoder_output = Conv2D(2, (1, 1), activation='tanh', padding='same')(decoder_c1)
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[CategoricalAccuracy()])
    return model


# 我電腦的GPU跑不動QQ
def best_version():
    encoder_input = Input(shape=(256, 256, 1,))

    encoder_c1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_input)
    encoder_c1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_c1)
    encoder_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c1)

    encoder_c2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_p1)
    encoder_c2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_c2)
    encoder_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c2)

    encoder_c3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_p2)
    encoder_c3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_c3)
    encoder_p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c3)

    encoder_c4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_p3)
    encoder_c4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_c4)
    encoder_p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c4)

    middle_in = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(encoder_p4)
    middle_in = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(middle_in)
    embed_input = Input(shape=(1000,))
    fusion_output = RepeatVector(16 * 16)(embed_input)
    fusion_output = Reshape(([16, 16, 1000]))(fusion_output)
    fusion_output = concatenate([middle_in, fusion_output], axis=3)
    fusion_output = Conv2D(512, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(fusion_output)
    middle_out = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(fusion_output)

    decoder_c4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(middle_out)
    decoder_c4 = concatenate([decoder_c4, encoder_c4])
    decoder_c4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c4)
    decoder_c4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c4)

    decoder_c3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(decoder_c4)
    decoder_c3 = concatenate([decoder_c3, encoder_c3])
    decoder_c3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c3)
    decoder_c3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c3)

    decoder_c2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(decoder_c3)
    decoder_c2 = concatenate([decoder_c2, encoder_c2])
    decoder_c2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c2)
    decoder_c2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c2)

    decoder_c1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(decoder_c2)
    decoder_c1 = concatenate([decoder_c1, encoder_c1])
    decoder_c1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c1)
    decoder_c1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(decoder_c1)

    decoder_output = Conv2D(2, (1, 1), activation='tanh', padding='same')(decoder_c1)
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[CategoricalAccuracy()])
    return model
