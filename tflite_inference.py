"""
The implementation of some  TFLite Inference for Semantic Segmentation.

@Author: Akash Yadav
@Github: https://github.com/ayadav10491
@Project: https://github.com/ayadav10491/TFLite-Semantic-Segmentation

"""




import tensorflow as tf
import numpy as np
#import pyrealsense2 as rs
import cv2 
import argparse
#import helpers
import sys
#import tflite_runtime.interpreter as tflite
from keras_applications import imagenet_utils

from utils.helpers import get_colored_info, color_encode
from utils.utils import load_image, decode_one_hot
from PIL import Image
import os
import argparse
import time

#np.set_printoptions(threshold=sys.maxsize)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--model', help='Choose the backbone model.', type=str, default="Saved_TFLite_Model/CamVid_Dataset_UNet_FLOAT16.tflite")
parser.add_argument('--csv_file', help='The path of color code csv file.', type=str, default="CamVid/class_dict.csv")
#parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, required=True)
#parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
#parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--test_image_path', help='The path of image to be predicted.', type=str, default="test_images")
parser.add_argument('--color_encode', help='Whether to color encode the prediction.', type=str2bool, default=True)
parser.add_argument('--save_path', help='The path of weights to be loaded.', type=str, default='CamVid_tflite_inference')

args = parser.parse_args()
#paths = check_related_path(os.getcwd())

if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)



if not os.path.exists(args.test_image_path):
    raise ValueError('The path \'{image_path}\' does not exist the image file.'.format(image_path=args.test_image_path))


# get color info
if args.csv_file is None:
    csv_file = os.path.join('CamVid', 'class_dict.csv')
else:
    csv_file = args.csv_file

_, color_values = get_colored_info(csv_file)

############ Load TFLite model and allocate tensors.      ##########################

interpreter = tf.lite.Interpreter(model_path=args.model)
print('Interpreter Details')
print(interpreter)

interpreter.allocate_tensors()

################# Get input and output tensors from interpreter. #####################################

input_details = interpreter.get_input_details()

print('Interpreter Input Details')
print(input_details)

output_details = interpreter.get_output_details()

print('Interpreter Output Details')
print(output_details)

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# load_images
image_names=list()
if os.path.isfile(args.test_image_path):
    image_names.append(args.test_image_path)
else:
    for f in os.listdir(args.test_image_path):
        image_names.append(os.path.join(args.test_image_path, f))
    image_names.sort()

frame=0
st=time.time()
for i, name in enumerate(image_names):
    t = time.time()
    sys.stdout.write('\rRunning test image %d / %d'%(i+1, len(image_names)))
    sys.stdout.flush()
    frame=frame+1
  
    image = cv2.resize(load_image(name),dsize=(width, height))
    color_image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last', mode='torch')

    if np.ndim(color_image) == 3:
        color_image = np.expand_dims(color_image, axis=0)
    assert np.ndim(color_image) == 4

    input_test_data = interpreter.set_tensor(input_details[0]['index'], color_image)
    #print(input_test_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output Data Shape", output_data.shape)

    if np.ndim(output_data) == 4:
                  prediction = np.squeeze(output_data, axis=0)

    # decode one-hot
    prediction = decode_one_hot(prediction)

    print('Starting Color Encoding')
    # color encode
    if args.color_encode:
            prediction = color_encode(prediction, color_values)
           


    # get PIL file
    prediction = Image.fromarray(np.uint8(prediction))

    # save the prediction
    _, file_name = os.path.split(name)
    print(file_name)
    prediction.save(os.path.join(args.save_path, file_name))
    time_ = time.time()-t
    print(time_)


predict_time=time.time()-st
m, s = divmod(predict_time, 60)
h, m = divmod(m, 60)

if s!=0:
  prediction_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
  print(prediction_time)
  fps= frame/predict_time
  print(fps)

else:
  print("No time")






