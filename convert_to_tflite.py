import os
import argparse
import tensorflow as tf
from tensorflow.keras import models
from tensorflow import lite
from custom_layer import ResizeImages

parser = argparse.ArgumentParser()
parser.add_argument('--input_keras_file', type=str)
parser.add_argument('--output_tflite_file', type=str)
FLAGS = parser.parse_args()

def main(argv=None):
    custom_objects = {
        "ResizeImages": ResizeImages
    }
    src = models.load_model(FLAGS.input_keras_file,
                            custom_objects=custom_objects,
                            compile=False)
    models.save_model(src, FLAGS.input_keras_file + '.freeze.h5', include_optimizer=False)
    dst = lite.TFLiteConverter.from_keras_model_file(FLAGS.input_keras_file + '.freeze.h5',
                                                     input_arrays=["input_image"],
                                                     input_shapes={"input_image":[1, 224, 224, 3]},
                                                     custom_objects=custom_objects).convert()
    with open(FLAGS.output_tflite_file, 'wb') as tflite_file:
        tflite_file.write(dst)
    interpreter = lite.Interpreter(model_content=dst)
    tensors = interpreter.get_tensor_details()
    for t in tensors:
        print(t)
        #print(interpreter.get_tensor(t['index']))

if __name__ == '__main__':
    main()
