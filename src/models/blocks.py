# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
segmentation and classification models builds
"""

import tensorflow as tf
import sys

def conv_block(inputs,
               conv_type="conv",
               filters=64,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding="same",
               relu=True,
               upsampling=False,
               up_sample_size=2,
               skip_layer=None
               ):
    """
    basic convolution block
    Args:
        inputs: input to the conv layer
        conv_type: type of convolution. "conv" for Conv2D and "ds" for depthwise separable convolution
        filters: number of filters
        kernel_size: kernel size used in convolution
        strides: strides used in convolution
        padding: padding used in convolution
        relu: if relu is active or not. 
        upsampling: if need to upsample
        skip_layer: if skip layer is added
    Returns:
        out: output of basic convolution block
    """
    if conv_type == "ds":
        out = tf.keras.layers.SeparableConv2D(filters=filters,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              strides=strides)(inputs)
    elif conv_type == "conv":
        out = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     strides=strides)(inputs)
    else:
        sys.exit("Wrong choice of convolution type.")
    
    out = tf.keras.layers.BatchNormalization()(out)
    
    if relu:
        out = tf.keras.activations.relu(out)
    
    if upsampling:
        out = tf.keras.layers.UpSampling2D(size=(up_sample_size, up_sample_size),
                                           data_format="channels_last")(out)
        skip_layer = tf.keras.layers.Conv2D(filters=out.shape[3], kernel_size=(1, 1), padding='same', strides=(1, 1))(skip_layer)
        out = tf.keras.layers.concatenate([out, skip_layer], axis=3)
        if conv_type == "ds":
            out = tf.keras.layers.SeparableConv2D(filters=filters,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                strides=strides)(out)
        elif conv_type == "conv":
            out = tf.keras.layers.Conv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        strides=strides)(out)
        else:
            sys.exit("Wrong choice of convolution type.")
        
        out = tf.keras.layers.BatchNormalization()(out)
        if relu:
            out = tf.keras.activations.relu(out)
        
    return out

def residual_bottleneck(inputs,
                        filters=64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        temp=3,
                        relu=False):
    """
    residual convolution bottleneck block
    Args:
        inputs: input to the bottleneck residual block
        filters: number of filters
        kernel_size: kernel size used in convolution
        temp: used in convolution
        relu: if relu is active or not. 
    Returns:
        out: output from residual convolution bottleneck block
    """
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * temp
    out = conv_block(inputs,
                   conv_type="conv",
                   filters=tchannel,
                   kernel_size=(1, 1))
    out = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                          depth_multiplier=1,
                                          strides=strides,
                                          padding="same")(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.activations.relu(out)
    out = conv_block(out,
                     conv_type="conv",
                     filters=filters,
                     kernel_size=(1, 1),
                     padding="same",
                     relu=False
    )
    if relu:
        out = tf.keras.layers.add([out, inputs])
    return out

def bottleneck_block(inputs,
                     filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     temp=6,
                     loop=3):
    """
    bottleneck block for residual convolution
    Args:
        inputs: input to the bottleneck residual block
        filters: number of filters
        kernel_size: kernel size used in convolution
        temp: used in convolution
        loop: number of times residual bottleneck need to apply
    Returns:
        out: output from residual bottleneck
    """
    out = residual_bottleneck(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            temp=temp)
    for _ in range(1, loop):
        out = residual_bottleneck(inputs=out,
                                filters=filters,
                                kernel_size=kernel_size,
                                temp=temp,
                                relu=True)
    return out

def pyramid_pooling_block(input_tensor, bin_sizes):
    """
    pyramid pooling block from PSPNet
    Args:
        input_tensor: input to the pyramid pooling block
        bin_sizes: bin used to create pyramid
    Returns:
        concat_list: output from pyramid pooling block
    """
    concat_list = [input_tensor]
    width = 16
    height = 16
    for bin_size in bin_sizes:
        out = tf.keras.layers.AveragePooling2D(
            pool_size=(width // bin_size, height // bin_size),
            strides=(width // bin_size, height // bin_size),
        )(input_tensor)
        out = tf.keras.layers.Conv2D(64, 3, 2, padding="same")(out)
        out = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (width, height)))(out)
        concat_list.append(out)
    return tf.keras.layers.concatenate(concat_list)

def tem_block(inputs):
    """
    Texture Enhancement Module
    Args:
        inputs: input to the tem module
    Returns:
        permute_2: output from the tem module
    """
    conv_1 = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=(3, 3),
                                    padding="same",
                                    name="conv_1")(inputs)
    gap_1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True,
                                                   name="gap_1")(conv_1)
    conv_2 = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=(1, 1),
                                    activation="relu",
                                    padding="same",
                                    name="conv_2")(gap_1)
    conv_3 = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=(1, 1),
                                    activation="sigmoid",
                                    padding="same",
                                    name="conv_3")(conv_2)
    mult_1 = tf.keras.layers.Multiply(name="mult_1")([conv_1, conv_3])
    add_1 = tf.keras.layers.Add(name="add_123")([conv_1, mult_1])

    gap_2 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True,
                                                   name="gap_2")(add_1)
    cos_sim_1 = tf.keras.layers.Dot(axes=(3),
                                    normalize=True,
                                    name="cos_sim1")([gap_2, add_1])
    reshape_1 = tf.keras.layers.Flatten()(cos_sim_1)
    transpose_1 = tf.keras.backend.transpose(reshape_1)
    mat_mul_1 = tf.keras.backend.dot(transpose_1,
                                     reshape_1)

    conv_4 = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=(1, 1),
                                    padding="same",
                                    name="conv_4")(add_1)
    permute_1 = tf.keras.layers.Permute((3, 1, 2),
                                        name="permute_1")(conv_4)
    reshape_2 = tf.keras.layers.Reshape((-1,
                                         int(conv_4.shape[-2]) * int(conv_4.shape[-3])),
                                         name="reshape_2")(permute_1)
    mat_mul_2 = tf.keras.backend.dot(reshape_2,
                                     mat_mul_1)

    reshape_3 = tf.keras.layers.Reshape((-1,
                                         int(conv_4.shape[-2]),
                                         int(conv_4.shape[-3])), name="reshape_3")(mat_mul_2)
    permute_2 = tf.keras.layers.Permute((2, 3, 1),
                                        name="permute_2")(reshape_3)

    return permute_2

def learning_module(input_layer):
    """
    Learning module
    Args:
        input_layer: input to the leaning module
    Returns:
        learning_layer3: output of learning module
    """
    learning_layer1 = conv_block(input_layer,
                                 conv_type="conv",
                                 filters=16,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 padding="same",
                                 relu=True)
    learning_layer2 = conv_block(learning_layer1,
                                 conv_type="ds",
                                 filters=32,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 padding="same",
                                 relu=True)
    learning_layer3 = conv_block(learning_layer2,
                                 conv_type="ds",
                                 filters=48,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 padding="same",
                                 relu=True)

    return learning_layer1, learning_layer2, learning_layer3

def feature_extractor(input_layer):
    """
    Feature Extractor module
    Args:
        input_layer: input to the Feature Extractor module
    Returns:
        fe_layer4: output of Feature Extractor module
    """
    fe_layer1 = bottleneck_block(input_layer,
                                 filters=48,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 temp=6,
                                 loop=3)
    fe_layer2 = bottleneck_block(fe_layer1,
                                 filters=64,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 temp=6,
                                 loop=3)
    fe_layer3 = bottleneck_block(fe_layer2,
                                 filters=96,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 temp=6,
                                 loop=3)
    fe_layer4 = pyramid_pooling_block(input_tensor=fe_layer3,
                                      bin_sizes=[2, 4, 6, 8])

    return fe_layer4

def fusion_module(learning_layer, fe_layer):
    """
    Feaure fusion module between learning module and feature extractor module outputs:
    Args:
        learning_layer: output of learning module
        fe_layer: output of feature extracted module
    Returns:
        fusion_layer: output of feature fusion module
    """
    fusion_layer1 = conv_block(learning_layer,
                               conv_type="conv",
                               filters=96,
                               kernel_size=(1, 1),
                               padding="same",
                               strides=(1, 1),
                               relu=True)

    fusion_layer2 = tf.keras.layers.UpSampling2D((4, 4))(fe_layer)
    fusion_layer2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                    strides=(1, 1),
                                                    depth_multiplier=1,
                                                    padding="same")(fusion_layer2)
    fusion_layer2 = tf.keras.layers.BatchNormalization()(fusion_layer2)
    fusion_layer2 = tf.keras.activations.relu(fusion_layer2)
    fusion_layer2 = tf.keras.layers.Conv2D(filters=96,
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           activation=None)(fusion_layer2)

    fusion_layer = tf.keras.layers.add([fusion_layer1, fusion_layer2])
    fusion_layer = tf.keras.layers.BatchNormalization()(fusion_layer)
    fusion_layer = tf.keras.activations.relu(fusion_layer)

    return fusion_layer

def get_encoder(image_height, image_width):
    """
    Helper functionn to create encoder of the model
    """
    input_layer = tf.keras.layers.Input(
        shape=(image_height, image_width, 3), name="input_layer"
    )

    # learning module
    learning_layer1, learning_layer2, learning_layer3 = learning_module(input_layer=input_layer)

    # feature extractor module
    fe_layer = feature_extractor(learning_layer3)

    # feature fusion module
    fusion_layer = fusion_module(learning_layer=learning_layer3, fe_layer=fe_layer)

    # build model
    model = tf.keras.Model(inputs=input_layer, outputs=fusion_layer, name="pcb-encoder")
    return model, learning_layer1, learning_layer2

def get_decoder(encoder, learning_layer1, learning_layer2, num_classes):

    inputs = encoder.input
    out = encoder.output
    out = tem_block(out)

    classifier1 = conv_block(out,
                             conv_type="ds",
                             filters=128,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding="same",
                             relu=True,
                             upsampling=True,
                             up_sample_size=2,
                             skip_layer=learning_layer2)
    
    classifier2 = conv_block(classifier1,
                             conv_type="ds",
                             filters=128,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding="same",
                             relu=True,
                             upsampling=True,
                            up_sample_size=2,
                            skip_layer=learning_layer1)
    
    
    classifier2 = tf.keras.layers.UpSampling2D(size=(2, 2),
                                       data_format="channels_last")(classifier2)

    classifier3 = tf.keras.layers.Conv2D(filters=num_classes,
                                         kernel_size=(1,1),
                                         padding="same",
                                         strides=(1, 1))(classifier2)
    classifier3 = tf.keras.layers.BatchNormalization(axis=3)(classifier3)
    classifier3 = tf.keras.activations.softmax(classifier3)

    # get final model
    model = tf.keras.models.Model(inputs=[inputs],
                                  outputs=[classifier3],
                                  name="pcb-decoder")
    return model

def get_classification(encoder, num_classes):
    inputs = encoder.inputs
    T1 = encoder.output
    gap = tf.keras.layers.GlobalAveragePooling2D()(T1)
    d1 = tf.keras.layers.Dense(128, activation="relu")(gap)
    d2 = tf.keras.layers.Dense(num_classes, activation="softmax")(d1)

    model = tf.keras.Model(inputs=inputs, outputs=d2, name="pcb-classification")
    return model
