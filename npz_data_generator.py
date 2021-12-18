"""
Based on the Pix2Pix GAN outlined by Jason Brownless
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

This helper script generates the .npz files required for the input of the cGAN
"""
from os import listdir
from numpy import asarray, vstack, savez_compressed
from keras.preprocessing.image import img_to_array, load_img


# load all images in a directory into memory
def load_images(path, size=(256, 256)):
    pic_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pic_list.append(img_to_array(pixels))
    return asarray(pic_list)


# path to synthetic snow images
path = 'center_crop_small/synthetic'
src_images = load_images(path)

# path to ground truth "no-snow" images
path = 'center_crop_small/gt'
tar_images = load_images(path)

print('Loaded: ', src_images.shape, tar_images.shape)

# save as compressed numpy array
filename = 'maps_256.npz' # save location
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)