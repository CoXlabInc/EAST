import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, ZeroPadding2D, Activation, Layer, Reshape
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.layers import InputSpec
import tensorflow as tf
import numpy as np
from custom_layer import ResizeImages

class EAST_model:

    def __init__(self, restore=''):
        input_size = 224
        input_image = Input(shape=(input_size, input_size, 3), name='input_image')

        weights = None
        if (restore is ''):
            weights = 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
            if not os.path.exists(weights):
                weights = 'imagenet'
                
        backbone = MobileNetV2(input_shape=(input_size, input_size, 3), alpha=0.5, input_tensor=input_image, weights=weights, include_top=False, pooling=None)

        x = backbone.get_layer('block_15_add').output

        shape = x.shape.as_list()
        x = ResizeImages(output_dim=(shape[1] * 2, shape[2] * 2))(x)
        
        x = concatenate([x, backbone.get_layer('block_12_add').output], axis=3, name='concatA')
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        shape = x.shape.as_list()
        x = ResizeImages(output_dim=(shape[1] * 2, shape[2] * 2))(x)
        
        x = concatenate([x, backbone.get_layer('block_5_add').output], axis=3, name='concatB')
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        shape = x.shape.as_list()
        x = ResizeImages(output_dim=(shape[1] * 2, shape[2] * 2))(x)
        
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

        if (restore is not ''):
            self.model.load_weights(restore)
