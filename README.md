# TFLite-Semantic-Segmantation
 Semantic Segmentation using TensorFlow Lite

![alt-text-10](https://github.com/ayadav10491/TFLite-Semantic-Segmantation/blob/master/utils/camvid.gif)

This work can be divided into two parts:
 
## Model (.h5) compression: 
   + This is true for all the trained models saved in .h5 formats irrespective of them being trained on semantic dataset or not. 
   + Model conversion uses TFLite Converter.
   + It encapsulates conversions of all the post-quatized formats i.e float16, float32, int8, tflite_without_optimization.
   + Use "Convert2TFLite.py" for the model stored in "weights/model.h5"
   + Some conversion formats need representative dataset as well which has been provided as well.
   
                            
## TFLite inference for semantic segmentation: 
   +  This process intiates the TFLite interpreter for the specific purpose the model was trained for (here CamVid Semantic Segmentation). 
   +  Use "tflite_inference.py" 
   +  please do not forget to set the path for the tflite model file. Currently it is set default inputs for outputs of "Convert2TFLite.py"
   
   
Credits: The CamVid semantic segmentation based model (.h5) is achieved from training the model using 
         'https://github.com/luyanger1799/Amazing-Semantic-Segmentation'


## Feedback
If you like this work, please give me a star! And if you find
any errors or have any suggestions, please contact me.  

**GitHub:** `ayadav10491`\
**Email:** `akashyadav10491@gmail.com` 
