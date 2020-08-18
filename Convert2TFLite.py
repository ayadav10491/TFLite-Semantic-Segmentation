"""
The implementation of some Kears Saved model to TFLite Conversion.

@Author: Akash Yadav
@Github: https://github.com/ayadav10491
@Project: https://github.com/ayadav10491/TFLite-Semantic-Segmentation

"""

import tensorflow as tf
import os
from tensorflow import keras
import argparse
from utils.helpers import get_dataset_info
from utils.utils import load_image
from keras_applications import imagenet_utils
import numpy as np
from utils.losses import categorical_crossentropy_with_logits

# tf.Enable_eager_execution()
import cv2

from keras.models import load_model


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', help='The path of the dataset.', type=str, default='CamVid')
parser.add_argument('--model_name', help='The path of the trained model.', type=str, default='./CamVid_UNet_based_on_MobileNetV2_crop_256_epochs_200_classes_32_batch_size_4_None.h5')
parser.add_argument('--model_quant_file', help='The path of the trained model.', type=str, default='CamVid_Dataset_UNet')

args = parser.parse_args()

saved_tflite_model_path = "Saved_TFLite_Model"

if not os.path.isdir(saved_tflite_model_path):
    os.makedirs(saved_tflite_model_path)

filename=os.path.join('weights',(args.model_name))

if os.path.isfile(filename):
    print('Model ', filename, ' exists')

    model = keras.models.load_model(filename, custom_objects={
        'categorical_crossentropy_with_logits': categorical_crossentropy_with_logits})

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
    converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
    converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name
    converter_float32_ = tf.lite.TFLiteConverter.from_keras_model(model)  # Your model's name

    train_image_names, train_label_names, valid_image_names, valid_label_names, _, _ = get_dataset_info(args.dataset)
    width=256
    height=256

    for i, name in enumerate(train_image_names):

            image=load_image(name)
            image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last',
                                                    mode='torch')
            image=cv2.resize(image,(width,height),interpolation=cv2.INTERSECT_NONE)
            image = np.expand_dims(image, axis=0)
            dataset_ = tf.data.Dataset.from_tensor_slices((image)).batch(1)

    def representative_data_gen():
                for input_value in dataset_.take(10):
                         yield [input_value]

   

    model_quant_file=args.model_quant_file
    
    #### INT8 QUANTIZATION #######

    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.target_spec.supported_types=[tf.compat.v1.lite.constants.FLOAT16]
    converter_int8.inference_input_type = tf.uint8
    converter_int8.inference_output_type = tf.uint8
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_gen
    tflite_model_quant_INT8 = converter_int8.convert()

    tflite_model_quant_file_INT8 = model_quant_file+'_INT8' +'.tflite'
    print(tflite_model_quant_file_INT8)
    tflite_model_quant_path_INT8 = os.path.join(saved_tflite_model_path,tflite_model_quant_file_INT8)
    print(tflite_model_quant_path_INT8)
    open(tflite_model_quant_path_INT8, "wb").write(tflite_model_quant_INT8)
    print('Conversion Successful. File written to ', tflite_model_quant_path_INT8)


    #### FLOAT32 QUANTIZATION #####

    converter_float32.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant_float32 = converter_float32.convert()
    tflite_model_quant_file_float32 = model_quant_file + '_FLOAT32' + '.tflite'
    print(tflite_model_quant_file_float32)
    tflite_model_quant_path_float32 = os.path.join(saved_tflite_model_path, tflite_model_quant_file_float32)
    print(tflite_model_quant_path_float32)
    open(tflite_model_quant_path_float32, "wb").write(tflite_model_quant_float32)
    print('Conversion Successful. File written to ', tflite_model_quant_path_float32)

    #### FLOAT16 QUANTIZATION #####

    converter_float16.target_spec.supported_types=[tf.compat.v1.lite.constants.FLOAT16]
    converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant_float16 = converter_float16.convert()
    tflite_model_quant_file_float16 = model_quant_file + '_FLOAT16' + '.tflite'
    print(tflite_model_quant_file_float16)
    tflite_model_quant_path_float16 = os.path.join(saved_tflite_model_path, tflite_model_quant_file_float16)
    print(tflite_model_quant_path_float16)
    open(tflite_model_quant_path_float16, "wb").write(tflite_model_quant_float16)
    print('Conversion Successful. File written to ', tflite_model_quant_path_float16)

    #### WITHOUT OPTIMIZATION  

    tflite_model_quant_float32_ = converter_float32_.convert()
    tflite_model_quant_file_float32_ = model_quant_file + '_without_quant' + '.tflite'
    print(tflite_model_quant_file_float32_)
    tflite_model_quant_path_float32_ = os.path.join(saved_tflite_model_path, tflite_model_quant_file_float32_)
    print(tflite_model_quant_path_float32_)
    open(tflite_model_quant_path_float32_, "wb").write(tflite_model_quant_float32_)
    print('Conversion Successful. File written to ', tflite_model_quant_path_float32_)
    


else:
    print('Some Error')
    exit()




