import os
from PIL import Image
from random import randrange

"""
Helper Script. Generates random crops from much larger images to be then fed into the smaller square input shape of
a GAN network. Setup to generate square cropped images from the snow100K dataset.
"""

crop_size = 64  # random crop size
num_samples = 10    # number of crops per image

# setup to generate random crops from both synthetic and ground truth directories of th snow100K dataset.
for gt, synth in zip(os.listdir("all/gt/"), os.listdir("all/synthetic/")):  # loops over both directories at same time
    # the above to file locations are where the original data is (relative path)
    # grab/open image
    gt_img = Image.open("all/gt/" + gt)
    synth_img = Image.open("all/synthetic/" + synth)

    # get image sizes
    x_gt, y_gt = gt_img.size
    x_synth, y_synth = synth_img.size

    # spot check they are the same (just in case)
    if(x_gt != x_synth or y_gt != y_synth or gt != synth):
        print("image size missmatch")
        break

    # perform random crop on each image
    for i in range(num_samples): # run for num_samples on each image
        x1 = randrange(0, x_gt - crop_size)
        y1 = randrange(0, y_gt - crop_size)
        save_gt = gt_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        save_synth = synth_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # save them bad bois
        save_gt.save("random_crop/gt/"+gt+"_"+str(i)+".jpg")
        save_synth.save("random_crop/synthetic/"+synth+"_"+str(i)+".jpg")

        # you will need the following file structure for output
        """
            /random crop
                /gt
                /synthetic
        
        """

