from numpy import load, zeros, ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, MaxPool2D, Dropout, BatchNormalization, LeakyReLU
from matplotlib import pyplot

"""
Based on the Pix2Pix GAN outlined by Jason Brownless
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/


Modifications have been made to test variations in;
1. Batch Size
2. Epoch Count
3. Downsampling
4. Inclusion of Batch Normalization
5. Dropout

This GAN is classified as a image translation conditional GAN or cGAN. The discriminator is provided with both
real and synthetic image and determined whether the synthetic is a plausible translation.
"""

# define the discriminator model
def define_discriminator(image_shape, batchnorm=True, conv_ds=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    # C64
    # Boolean decides between the use of simple 2D convolution or 2D convolution + Max Pooling
    if conv_ds:
        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    else:
        d = Conv2D(64, (4, 4), padding='same', kernel_initializer=init)(merged)
        d = MaxPool2D(pool_size=(2, 2), strides=2)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C128
    # Same boolean switch as above
    if conv_ds:
        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    else:
        d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init)(d)
        d = MaxPool2D(pool_size=(2, 2), strides=2)(d)
    # Boolean decides if batch normalization is applied between layers before Leaky Relu activation
    if batchnorm:
        d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256
    if conv_ds:
        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    else:
        d = Conv2D(256, (4, 4), padding='same', kernel_initializer=init)(d)
        d = MaxPool2D(pool_size=(2, 2), strides=2)(d)
    if batchnorm:
        d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C512
    if conv_ds:
        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    else:
        d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
        d = MaxPool2D(pool_size=(2, 2), strides=2)(d)
    if batchnorm:
        d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    if batchnorm:
        d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True, conv_ds=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    if conv_ds:
        g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    else:
        g = Conv2D(n_filters, (4, 4), padding='same', kernel_initializer=init)(layer_in)
        g = MaxPool2D(pool_size=(2, 2), strides=2)(g)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True, dropout_rate=0.5):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(dropout_rate)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 3), conv_ds=True, dropout_rate=0.5):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, False, conv_ds)
    e2 = define_encoder_block(e1, 128, True, conv_ds)
    e3 = define_encoder_block(e2, 256, True, conv_ds)
    e4 = define_encoder_block(e3, 512, True, conv_ds)
    e5 = define_encoder_block(e4, 512, True, conv_ds)
    e6 = define_encoder_block(e5, 512, True, conv_ds)
    e7 = define_encoder_block(e6, 512, True, conv_ds)

    # bottleneck, no batch norm and relu
    if conv_ds:
        b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    else:
        b = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(e7)
        b = MaxPool2D(pool_size=(2, 2), strides=2)(b)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512, True, dropout_rate)
    d2 = decoder_block(d1, e6, 512, True, dropout_rate)
    d3 = decoder_block(d2, e5, 512, True, dropout_rate)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)


###########################
# Experimental Parameters #
###########################

# Batch Size - 1 is base case
n_batch = 1

# Epoch Size - 100 is base case
n_epochs = 100

# Downsampling (Discriminator, Encoder and Bottleneck) - default is conv (True), False is Maxpool
conv_ds = True

# Batch Normalization (Discriminator only) - default is on (True), False is none
batchnorm = False

# Dropout (Generator only) - 0.5 is base case
dropout_rate = 0.5

###########################
# Experimental Parameters #
###########################

# load image data
print('Data loading started...')
dataset = load_real_samples('maps_256.npz') # local location of the file generated by the npz_data_generator.py
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape, batchnorm, conv_ds)
g_model = define_generator(image_shape, conv_ds, dropout_rate)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model

d_model.summary()
g_model.summary()

train(d_model, g_model, gan_model, dataset, n_epochs, n_batch)