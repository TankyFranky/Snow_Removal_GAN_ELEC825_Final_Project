import random
import os
import glob
from tqdm import tqdm
from PIL import Image

"""
Helper script. SnowyCeleb generates synthetic snowy data by using the masks provided in the Snow100K dataset
and any other input images. Setup for the celebA dataset and 64x64 sized images.
"""

# change \ to /, python takes \ as an escape character
MASK_DIRECTORY = "/mask" # directory of snow masks (.jpg from Snow100K dataset)
CELEBA_DIRECTORY = "/img_align_celeba" # directory of 64x64 CelebA face dataset (.jpg)
SAVE_DIRECTORY = "/SnowyCeleb" # directory where the new synthetic data will be saved too


style = 'random crop' # 'random crop' and 'resize' are supported

# for each celebA face
for celebFace in tqdm(glob.glob(CELEBA_DIRECTORY+"/*.jpg")):
    # compress to 64x64
    celebFace_PIL = Image.open(celebFace)
    celebFace_PIL = celebFace_PIL.resize((64,64))

    # get snow image
    snow_PIL = Image.open(MASK_DIRECTORY +"/"+ random.choice(os.listdir(MASK_DIRECTORY))).convert("RGBA")
    snow_PIL.save("temp.png", format="png")

    # turn image in something with an alpha mask
    snow_PIL = Image.open("temp.png").convert('L')

    # choose style of snow application
    if style == 'resize':
        snow_PIL = snow_PIL.resize((64,64))

    if style == 'random crop':
        # crop a 64,64 mask
        x, y = snow_PIL.size

        x1 = random.randrange(0, x - 64)
        y1 = random.randrange(0, y - 64)
        snow_PIL = snow_PIL.crop((x1, y1, x1 + 64, y1 + 64))


    # apply random mask
    celebFace_PIL.paste(snow_PIL, (0,0), snow_PIL)

    # save image
    file_name = "snow_face_" + os.path.basename(celebFace)+".jpg"
    celebFace_PIL.save(SAVE_DIRECTORY + "/" + file_name)