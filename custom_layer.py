import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.layers import InputSpec

class ResizeImages(Layer):
    def __init__(self, output_dim=(1, 1), **kwargs):
        super(ResizeImages, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1], input_shape[3])

    def call(self, inputs):
        output = tf.image.resize_bilinear(inputs, size=self.output_dim)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(ResizeImages, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
