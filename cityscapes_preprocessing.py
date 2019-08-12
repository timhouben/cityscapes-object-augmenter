import os
import numpy as np
import cv2
import glob
import time
import datetime
from PIL import Image
from labelinfo import name2label

# ======================================
# parameters
# ======================================

# category to add (master (optional: followed by slaves))
#name = ['pole']
name = ['pole', 'traffic light', 'traffic sign']

# number of samples to be created (0 is maximum possible)
number_of_samples = 0

# number of source images to pick from
number_of_source_images = 500

# limits of number of new object to be placed
maximum_objects_min = 20
maximum_objects_max = 30

# minimum number of pixels the new object needs to have
minimum_object_size = 500

# allowed terrain to place objects on
allowed_terrain = ['ground', 'road', 'sidewalk', 'parking']

# forbidden classes to intersect for new objects
no_intersection = ['pole', 'traffic light', 'traffic sign']

# probability of placement to add more randomness
p_placement = 0.5

# image dimensions
img_h = 1024
img_w = 2048

# folder paths
folder_in = "D:\\final_assignment\\datasets\\gtFine_trainvaltest\\gtFine\\train\\"
folder_in_disparity = "D:\\final_assignment\\datasets\\disparity_trainvaltest\\disparity\\train\\"
folder_out = "D:\\final_assignment\\datasets\\testdata\\"

# ======================================
# begin of script
# ======================================


def list_images(path):
    return [f for f in glob.glob(path, recursive=True)]


def open_image(image):
    with open(image, 'rb') as file:
        return np.array(Image.open(file))


def resize_image(image):
    return cv2.resize(image.astype(np.float32), dsize=(64, 128), interpolation=cv2.INTER_CUBIC)


def no_intersect(cats, current, target):
    output = False
    for i in range(len(cats)):
        one_output = np.logical_and(current, np.equal(target, cats[i] * np.ones((img_h, img_w), dtype=np.int))).sum() == 0
        output = np.logical_or(output, one_output)
    return output


def get_label_id(cat_name):
    return name2label[cat_name].id


def get_label_ids(names):
    return list(map(lambda x: get_label_id(x), names))


def save_image(data, mode, ending, subfolder):
    new_image = Image.fromarray(data, mode)
    file_name = ("eindhoven_" + '{0:06d}'.format(n) + ending)
    path = os.path.join(folder_out, subfolder, file_name)
    return new_image.save(path, "PNG")


print('Initializing...')

# get class ids from names
mask_cat = np.empty((len(name), 1), dtype=np.int)
mask_color = np.empty((len(name), 4), dtype=np.int)
for n in range(len(name)):
    mask_cat[n, :] = get_label_id(name[n])
    mask_color[n, :] = np.append(np.array(name2label[name[n]].color), 0)

# get lists of all images available
images_source = list_images(folder_in + "**/*_gtFine_labelIds.png")
images_source_color = list_images(folder_in + "**/*_gtFine_color.png")
images_bg = list_images(folder_in + "**/*_gtFine_labelIds.png")
images_bg_inst = list_images(folder_in + "**/*_gtFine_instanceIds.png")
images_color = list_images(folder_in + "**/*_gtFine_color.png")
images_target_disparity = list_images(folder_in_disparity + "**/*_disparity.png")
images_source_disparity = list_images(folder_in_disparity + "**/*_disparity.png")

# shuffle for the background
indices_bg = np.random.permutation(len(images_bg))
images_target = np.array(images_bg)[indices_bg.astype(int)]
images_target_inst = np.array(images_bg_inst)[indices_bg.astype(int)]
images_target_color = np.array(images_color)[indices_bg.astype(int)]
images_target_disparity = np.array(images_target_disparity)[indices_bg.astype(int)]

# truncate if needed
if number_of_samples != 0:
    images_target = images_target[0:number_of_samples]
    images_target_inst = images_target_inst[0:number_of_samples]
    images_target_color = images_target_color[0:number_of_samples]
    images_target_disparity = images_target_disparity[0:number_of_samples]
else:
    number_of_samples = len(images_source)

# create the masks
mask_operator = np.empty((len(name), img_h, img_w), dtype=np.int)
mask_operator_color = np.empty((len(name), img_h, img_w, 4), dtype=np.int)
for n in range(len(name)):
    mask_operator[n, :, :] = np.multiply(mask_cat[n], np.ones((img_h, img_w), dtype=np.int))
    mask_operator_color[n, :, :, :] = np.multiply(mask_color[n], np.ones((img_h, img_w, 4), dtype=np.int))

# start the loop (1 new sample per iteration)
print('Start the loop...')
for n in range(number_of_samples):

    # open target image
    target_image = open_image(images_target[n])
    # open background instance image
    target_inst_image = open_image(images_target_inst[n])
    # open color image
    target_color_image_orig = open_image(images_target_color[n])
    target_color_image = open_image(images_target_color[n])
    # open disparity image
    target_disparity_image = open_image(images_target_disparity[n])

    # determine maximum objects for this image
    maximum_objects = np.random.randint(maximum_objects_min, maximum_objects_max)

    # random shuffle to pick objects
    indices_obj = np.random.permutation(len(images_source))
    images_source = np.array(images_source)[indices_obj.astype(int)]
    images_source_color = np.array(images_source_color)[indices_obj.astype(int)]
    images_source_disparity = np.array(images_source_disparity)[indices_obj.astype(int)]

    # truncate
    images_source_shortlist = images_source[0:number_of_source_images]
    images_source_color_shortlist = images_source_color[0:number_of_source_images]
    images_source_disparity_shortlist = images_source_disparity[0:number_of_source_images]

    # images with similar depth maps first
    target_disparity_image = resize_image(target_disparity_image)
    distances = np.zeros(number_of_source_images, dtype=np.int)
    for m in range(number_of_source_images):
        current_disparity_image = open_image(images_source_disparity_shortlist[m])
        current_disparity_image = resize_image(current_disparity_image)
        # calculate euclidean distance
        distances[m] = np.sqrt(np.power(np.subtract(target_disparity_image, current_disparity_image), 2).sum())
    new_indices = np.argsort(distances, axis=0)
    images_source_shortlist = images_source_shortlist[new_indices.astype(int)]
    images_source_color_shortlist = images_source_color_shortlist[new_indices.astype(int)]

    # for all source images
    object_counter = 0
    custom_mask = np.zeros((3, img_h, img_w), dtype=np.uint8)
    for m in range(number_of_source_images):
        # open source image
        source_image = open_image(images_source_shortlist[m])
        # open source image color
        source_color_image = open_image(images_source_color_shortlist[m])

        # create mask and combined mask of the objects to be placed
        object_mask = np.zeros((3, img_h, img_w), dtype=np.uint8)
        combined_object_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for k in range(len(name)):
            object_mask[k, :, :] = np.equal(source_image, mask_operator[k]).astype(np.uint8)
            combined_object_mask = np.add(combined_object_mask, np.equal(source_image, mask_operator[k]).astype(np.uint8))

        # create instances and perform checks individually
        ret, labels = cv2.connectedComponents(combined_object_mask)

        # for every object group
        for l in range(ret-1):
            # find the lowest pixel of the pole
            lowest_pixel = np.array(np.where(labels == l+1))[:, -1]
            # check where the pole will be placed in the target image
            ground_category = target_image[lowest_pixel[0], lowest_pixel[1]]

            current_object_mask = np.equal(labels, (l + 1) * np.ones((img_h, img_w))).astype(np.uint8)
            # perform several checks if selected object is valid for placement
            if all([# check if no intersection with other pole or signs
                    no_intersect(get_label_ids(no_intersection), current_object_mask, target_image),
                    # check if pole will be placed on ground road or sidewalk
                    ground_category in get_label_ids(allowed_terrain),
                    # check if object is big enough to place
                    current_object_mask.sum() > minimum_object_size,
                    # add randomness
                    np.random.uniform(0, 1) > p_placement]):
                # found an object!
                object_counter += 1
                for k in range(len(name)):
                    filter_category = np.logical_and(current_object_mask, object_mask[k, :, :]).astype(np.uint8)
                    custom_mask[k, :, :] = np.add(custom_mask[k, :, :], filter_category).astype(np.uint8)

        # when enough objects are added to the mask
        if object_counter > maximum_objects:

            # create inverted mask
            custom_mask = np.clip(custom_mask, 0, 1)
            inverted_custom_mask = np.subtract(1, custom_mask).astype(np.uint8)

            # separate for every object category
            for k in range(len(name)):
                # remove pixels where new objects will be placed
                target_image = np.multiply(inverted_custom_mask[k, :, :], target_image).astype(np.uint8)
                # add new objects to background
                cat_mask = np.multiply(mask_cat[k], custom_mask[k, :, :])
                target_image = np.add(target_image, cat_mask).astype(np.uint8)

                # for each channel
                for l in range(3):
                    # remove pixels where new objects will be placed
                    target_color_image[:, :, l] = np.multiply(inverted_custom_mask[k, :, :], target_color_image[:, :, l])
                    # add new objects to colormap
                    cat_color_mask = np.multiply(mask_color[k, l], custom_mask[k, :, :])
                    target_color_image[:, :, l] = np.add(target_color_image[:, :, l], cat_color_mask).astype(np.uint8)

                # remove possible pixels from instance maps
                target_inst_image = np.multiply(inverted_custom_mask[k, :, :], target_inst_image).astype(np.int32)

            break

    # save new id image to disk
    save_image(target_image, 'L', "_000019_gtFine_labelIds.png", "gtFine\\")
    # save new instance image to disk
    save_image(target_inst_image, 'I', "_000019_gtFine_instanceIds.png", "gtFine\\")
    # save new color image to disk
    save_image(target_color_image, 'RGBA', "_000019_gtFine_color.png", "gtFine\\")
    # save original color image to disk (separate folder)
    save_image(target_color_image_orig, 'RGBA', "_000018_gtFine_color.png", 'gtFineOrig\\')
    # save dummy 8bit image
    save_image(np.zeros((img_h, img_w, 3), dtype=np.uint8), 'RGB', "_000019_leftImg8bit.png", "leftImg8bit\\")

    # print progress
    ts = time.time()
    current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(str(current_time) + ': ' + str(n+1) + '/' + str(number_of_samples))

print('Done!')