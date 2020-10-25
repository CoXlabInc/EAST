from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, ZeroPadding2D, Activation, Layer, Reshape
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.layers import InputSpec
import tensorflow as tf
import numpy as np

RESIZE_FACTOR = 2

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

def resize_bilinear(x):
    return tf.image.resize_bilinear(x, size=[K.shape(x)[1]*RESIZE_FACTOR, K.shape(x)[2]*RESIZE_FACTOR])

def resize_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= RESIZE_FACTOR
    shape[2] *= RESIZE_FACTOR
    return tuple(shape)

class EAST_model:

    def __init__(self, input_size=512):
        input_image = Input(shape=(224, 224, 3), name='input_image')
        backbone = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, input_tensor=input_image, weights='pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5', include_top=False, pooling=None)
        x = backbone.get_layer('block_15_add').output

        x = ResizeImages(output_dim=(14, 14))(x)
        x = concatenate([x, backbone.get_layer('block_12_add').output], axis=3, name='concatA')
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = ResizeImages(output_dim=(28, 28))(x)
        x = concatenate([x, backbone.get_layer('block_5_add').output], axis=3, name='concatB')
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = ResizeImages(output_dim=(56, 56))(x)
        #x = concatenate([x, ZeroPadding2D(((1, 0),(1, 0)))(backbone.get_layer('block_2_add').output)], axis=3, name='concatC')
        x = concatenate([x, backbone.get_layer('block_2_add').output], axis=3, name='concatC')
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x) 
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        self.model_core = Model(inputs=input_image, outputs=[pred_score_map, pred_geo_map])
        
        overly_small_text_region_training_mask = Input(shape=(None, None, 1), name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(None, None, 1), name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(None, None, 1), name='target_score_map')
        self.model = Model(inputs=[self.model_core.input, overly_small_text_region_training_mask, text_region_boundary_training_mask, target_score_map], outputs=self.model_core.output)

        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map
        self.backbone = backbone
