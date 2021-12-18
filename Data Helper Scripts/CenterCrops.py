import os

"""
Helper Script. Creates 256x256 center cropped images from the snow100K dataset. Based on the GenerateCrops.py script
"""

crop_size = 256 # crop size
num_samples = 1 # number of crops per image
counter = 0

for gt, synth in zip(os.listdir("all/gt/"), os.listdir("all/synthetic/")):  # loops over both directories at same time
#for realistic in os.listdir("Dataset/Snow100K/realistic/"):
    # the above to file locations are where the original data is (relative path)
    # grab/open image
    gt_img = Image.open("Dataset/Snow100K/test/Snow100K-L/gt/" + gt)
    synth_img = Image.open("Dataset/Snow100K/test/Snow100K-L/synthetic/" + synth)
    #realistic_img = Image.open("Dataset/Snow100K/realistic/" + realistic)

    # get image sizes
    x_gt, y_gt = gt_img.size
    x_synth, y_synth = synth_img.size
    #x_realistic, y_realistic = realistic_img.size

    #close image to avoid errors
#     synth_img.close()

#     # remove images less than 256x256
#     if (x_synth < 256):
#         counter = counter + 1
        #os.remove("Dataset/Snow100K/test/Snow100K-L/synthetic/" + synth)

#print(counter)

    #spot check they are the same (just in case)
#     if(x_gt != x_synth or y_gt != y_synth or gt != synth):
#         print("image size missmatch")
#         break

    # perform center crop on each image
    # run for num_samples on each image
    for i in range(num_samples): # run for num_samples on each image
        x1 = (x_synth - crop_size)/2
        y1 = (y_synth - crop_size)/2
        save_gt = gt_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        save_synth = synth_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # save them bad bois
        save_gt.save("Dataset/Snow100K/realistic_crop/"+gt+"_"+str(i)+".jpg")
        save_synth.save("Dataset/Snow100K/center_crop_small_val/synthetic/"+synth+"_"+str(i)+".jpg")

        # you will need the following file structure for output
        """
            /center crop
                /gt
                /synthetic
        
        """