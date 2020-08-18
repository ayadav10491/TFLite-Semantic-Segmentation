"""
The implementation of some Kears Saved model to TFLite Conversion.

@Author: Akash Yadav
@Github: https://github.com/ayadav10491
@Project: https://github.com/ayadav10491/TFLite-Semantic-Segmentation

"""


from utils.utils import load_image, decode_one_hot
from keras_preprocessing import image as keras_image
from keras_applications import imagenet_utils

from PIL import Image
import numpy as np
import cv2
import os
import argparse
import sys
import time


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
parser.add_argument('--seg_image_path', help='Segmented image to be predicted.', type=str, default="CamVid_tflite_inference")

parser.add_argument('--color_encode', help='Whether to color encode the prediction.', type=str2bool, default=True)
parser.add_argument('--save_path_mask', help='The path of weights to be loaded.', type=str, default='CamVid_tflite_inference_mask')

args = parser.parse_args()
#paths = check_related_path(os.getcwd())

if not os.path.isdir(args.save_path_mask):
        os.makedirs(args.save_path_mask)



alpha = 0.5

"""
raw_input = input  # Python 3

"""
if not os.path.exists(args.test_image_path):
    raise ValueError('The path \'{image_path}\' does not exist the image file.'.format(image_path=args.test_image_path))

if not os.path.exists(args.test_image_path):
    raise ValueError('The path \'{seg_image_path}\' does not exist the image file.'.format(seg_image_path=args.seg_image_path))

print('Here1')
# load_images
test_image_names=list()
if os.path.isfile(args.test_image_path):
    test_image_names.append(args.test_image_path)
else:
    for f in os.listdir(args.test_image_path):
        test_image_names.append(os.path.join(args.test_image_path, f))
    test_image_names.sort()

# load_segmented_images
seg_image_names=list()
if os.path.isfile(args.seg_image_path):
    image_names.append(args.seg_image_path)
else:
    for j in os.listdir(args.seg_image_path):
        seg_image_names.append(os.path.join(args.seg_image_path, j))
    seg_image_names.sort()

"""
input_alpha = float(raw_input().strip())

if 0 <= alpha <= 1:
    alpha = input_alpha
"""
# [load]
frame=0
st=time.time()

for i, test_name in enumerate(test_image_names):
    t = time.time()
    sys.stdout.write('\rRunning test image %d / %d'%(i+1, len(test_image_names)))

    sys.stdout.flush()
    frame=frame+1

    width=256
    height=256
  
    test_image = cv2.resize(load_image(test_name),dsize=(width, height))
    print(test_image.dtype)
    #color_image = imagenet_utils.preprocess_input(test_image.astype(np.float32), data_format='channels_last', mode='torch')
    

    #print(color_image.dtype)

    retrieve_name= os.path.split(test_name)

    seg_name = os.path.join(args.seg_image_path, retrieve_name[1])
    print(seg_name)

    seg_image = load_image(seg_name)
    print(seg_image.dtype)
    #seg_color_image = imagenet_utils.preprocess_input(seg_image.astype(np.float32), data_format='channels_last', mode='torch')
    #seg_color_image =seg_image.astype(np.float32)
    #seg_color_image /= 255

    # print(seg_color_image.dtype)
     
    # [blend_images]
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(test_image, alpha, seg_image, beta, 0.0)
    print(dst.dtype)
    
    save_file_path =os.path.join(args.save_path_mask, retrieve_name[1])
    print(save_file_path)
    cv2.imwrite(save_file_path, cv2.cvtColor(dst,cv2.COLOR_RGB2BGR))

    #Save Images

    
# [display]
