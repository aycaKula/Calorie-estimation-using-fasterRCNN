import tkinter
import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import pathlib
import glob
import matplotlib.pyplot as plt
import cv2
import math
import csv
import openpyxl
import pandas as pd

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# types of food in the dataset
foods = ['mix', 'apple', 'egg', 'lemon', 'orange', 'peach', 'plum', 'qiwi', 'tomato', 'bread', 'grape', 'mooncake',
        'sachima', 'banana', 'bun', 'doughnut', 'fired_dough_twist', 'litchi', 'mango', 'pear']


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# radius of coin
rad_coin = 2.5/2
# area of the coin cm^2
coin_area = math.pi* rad_coin**2

food_id_name = []
label_new = []
estimated_volu = []


def get_volume(food_set_name, side_image_prop, top_image_prop):
    '''
    Get volume for the types of foods
    Args:
        side_image_prop: side image dictionary
        top_image_prop: top image dictionary

    Returns: Return each objects volume

    '''
    estimated_volume = {}
    side_img_labels = list(side_image_prop)
    top_img_labels = list(top_image_prop)

    num = 0

    #if set(side_img_labels) != set(top_img_labels):
        #check_match = food_set_name + 'Labels does not match !!!!!!'
        #print(check_match)
        #num = 1

    if num == 0:
        for label in side_img_labels:
            if label != 'coin': # You do not have to calculate volume for coin
                # if shape is sphere
                if label == 'apple' or label == 'litchi' or label == 'orange' or label == 'pear':
                    if side_image_prop['coin'][0] != 0:
                        side_area = float(side_image_prop[label][0])*coin_area/float(side_image_prop['coin'][0])
                        radius = math.sqrt(side_area/math.pi)
                        estimated_vol = (4/3)*math.pi*radius**3
                    else:
                        estimated_vol = 0
                # if the shape is a column
                elif label == 'banana' or label == 'doughnut' or label == 'bread' or label == 'bun' or label == 'fired_dough_twist' or label == 'grape' or label == 'mooncake' or label == 'sachima':
                    # depth measurement with respect to coin pixel/diameter relation
                    if side_image_prop['coin'][2] !=0 and top_image_prop['coin'][0] !=0:
                        side_height = 2*rad_coin*side_image_prop[label][2]/side_image_prop['coin'][2]
                        estimated_vol = float(top_image_prop[label][0])*coin_area/float(top_image_prop['coin'][0])*side_height
                    else:
                        estimated_vol = 0
                # if shape is ellipsoid:
                elif label == 'egg' or label == 'lemon' or label == 'mango' or label== 'peach' or label == 'plum' or label == 'qiwi' or label == 'tomato':
                    if side_image_prop['coin'][2] !=0 and top_image_prop['coin'][1]!=0 and top_image_prop['coin'][2]!=0:
                        side_height = 2 * rad_coin * side_image_prop[label][2] / side_image_prop['coin'][2]
                        top_width = 2 * rad_coin * side_image_prop[label][1] / top_image_prop['coin'][1]
                        top_height = 2 * rad_coin * side_image_prop[label][2] / top_image_prop['coin'][2]
                        a = side_height/2
                        b = top_height/2
                        c = top_width/2
                        estimated_vol = (4 / 3)*math.pi*a*b*c
                    else:
                        estimated_vol = 0

                estimated_volume[label] = estimated_vol
                food_id_name.append(food_set_name)
                label_new.append(label)
                estimated_volu.append(estimated_vol)
    else:
        estimated_volume['error'] = 1

    # label, volume estimation in new dic
    new_dict = {'id': food_id_name, 'label': label_new, 'Volume est': estimated_volu}
    df = pd.DataFrame(new_dict)
    df.to_csv('fil.csv')

    return estimated_volume, num


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.
  Args:
    path: a file path (this can be local or on colossus)
  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.75, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, image_path):
    '''

    Args:
        model:
        category_index:
        image_path: path of image

    Returns:
        Bounding boxes and segmented objects
    '''
    if os.path.isdir(image_path):
        image_paths = []
        for file_extension in ('*.png', '*jpg'):
            image_paths.extend(glob.glob(os.path.join(image_path, file_extension)))

        for i_path in image_paths:
            image_np = load_image_into_numpy_array(i_path)
            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.imshow(image_np)
            plt.show()
    else:
        image_np = load_image_into_numpy_array(image_path)

        # Opens a image in RGB mode
        #im = Image.open(image_path)
        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
        im_width, im_height = image_pil.size
        #print(im_width)
        #print(im_height)
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            min_score_thresh=0.75,
            line_thickness=1)
            #skip_scores=True,
            #skip_labels=True)
        plt.imshow(image_np)
        plt.savefig('deneme.jpg')
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('deneme_wo_border.jpg', bbox_inches='tight', pad_inches=0)

        # This is the way I'm getting my coordinates
        boxes = output_dict['detection_boxes']
        # get all boxes from an array
        max_boxes_to_draw = boxes.shape[0]
        # get scores to get a threshold
        scores = output_dict['detection_scores']
        # this is set as a default but feel free to adjust it to your needs
        min_score_thresh = .9
        # iterate over all objects found
        main_dict = {}
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            #
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                class_name = category_index[output_dict['detection_classes'][i]]['name']
                # ymin, xmin, ymax, xmax = box
                #print("This box is gonna get used", boxes[i], output_dict['detection_classes'][i], output_dict['detection_scores'][i])

                (left, right, top, bottom) = (boxes[i, 1] * im_width, boxes[i, 3] * im_width, boxes[i, 0] * im_height, boxes[i, 2] * im_height)

                #print(left)
                #print(right)
                #print(top)
                #print(bottom)
                left = int(round(left))
                right = int(round(right))
                top = int(round(top))
                bottom = int(round(bottom))
                #print(image_np.shape)
                #(left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height)
                #crop_img = tf.image.crop_to_bounding_box(image_np, top-bottom, right-left, top, right)
                #mask image
                mask = np.zeros(image_np.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                bound_box_width = right - left
                bound_box_height = bottom - top
                rect = (left+4, top+2, right-left-8, bottom-top-4)
                # Apply Grab-Cut Segmentation Algorithm
                cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                imgMask = image_np * mask2[:, :, np.newaxis]
                grayMask = cv2.cvtColor(imgMask, cv2.COLOR_BGR2GRAY)
                main_dict[class_name] = [cv2.countNonZero(grayMask), bound_box_width, bound_box_height]
                plt.imshow(imgMask)
                plt.savefig(class_name+'_mask.jpg')
                # crop image
                # crop_img = image_np[(top-15):(bottom+15), (left-15):(right+15)]
                #plt.imshow(crop_img)
                #class_name = str(output_dict['detection_classes'][i]) + '.jpg'
                #plt.axis('off')
                #plt.gca().xaxis.set_major_locator(plt.NullLocator())
                #plt.gca().yaxis.set_major_locator(plt.NullLocator())
                #plt.savefig(class_name, bbox_inches='tight', pad_inches = 0)

    return main_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    dest = 'D:/PythonProjects/TensorFlow2TheFasterKing/models/research/object_detection/images/reduced_data'

    files = os.listdir(dest)
    new_name = []
    test_set = []
    volume = {}
    volume_id = {}
    side_image_prop = {}
    top_image_prop = {}
    for food_name in foods:
        for i_file in files:
            if food_name in i_file:
                new_name.append(i_file)
        food_length = len(new_name) / 2
        for i in range(int(food_length)):
            if i + 1 < 10:
                food_s = food_name + '00' + str(i + 1)
                for each_name in new_name:
                    if food_s in each_name:
                        test_set.append(each_name)

            else:
                food_s = food_name + '0' + str(i + 1)
                for each_name in new_name:
                    if food_s in each_name:
                        test_set.append(each_name)


            # İ_TEST --> index of the couple (top and side) set
            for i_test in range(len(test_set)):
                if i_test == 0:
                    side_image_path = 'images/reduced_data/' + test_set[i_test]
                    side_image_prop = run_inference(detection_model, category_index, side_image_path)
                    dataset_side = food_s + ' side_image_prop' + str(side_image_prop)
                    print(dataset_side)

                else:
                    top_image_path = 'images/reduced_data/' + test_set[i_test]
                    top_image_prop = run_inference(detection_model, category_index, top_image_path)
                    dataset_top = food_s + ' top_image_prop' + str(top_image_prop)
                    print(dataset_top)

            volume_dict, num = get_volume(food_s, side_image_prop, top_image_prop)
            print(volume_dict)

            # if all(x==0 for x in top_image_prop.values())==False or all(x==0 for x in side_image_prop.values())==False:
            #myworkbook = openpyxl.load_workbook('density.csv')
            #worksheet = myworkbook.get_sheet_by_name('Sheet1')

            #for ii in range(len(list(volume_dict))):
            #    # i: length of food
            #    id_index = 'A' + i + ii
            #    worksheet['B4'] = 'We are writing to B4'




            if num == 1:
                continue
            #else:
            #    food_cal = get_calorie()

            test_set = []
            side_image_prop = {}
            top_image_prop = {}
            volume_id = {}

        new_name = []
