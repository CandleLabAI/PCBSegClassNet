# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

import tensorflow as tf


def conv_block(
    inputs, conv_type, kernel, kernel_size, strides, padding="same", relu=True
):
    if conv_type == "ds":
        x = tf.keras.layers.SeparableConv2D(
            kernel, kernel_size, padding=padding, strides=strides
        )(inputs)
    else:
        x = tf.keras.layers.Conv2D(
            kernel, kernel_size, padding=padding, strides=strides
        )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if relu:
        x = tf.keras.activations.relu(x)
    return x


def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t
    x = conv_block(inputs, "conv", tchannel, (1, 1), strides=(1, 1))
    x = tf.keras.layers.DepthwiseConv2D(
        kernel, strides=(s, s), depth_multiplier=1, padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = conv_block(
        x, "conv", filters, (1, 1), strides=(1, 1), padding="same", relu=False
    )
    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)
    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)
    return x


def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    w = 16
    h = 16
    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(
            pool_size=(w // bin_size, h // bin_size),
            strides=(w // bin_size, h // bin_size),
        )(input_tensor)
        x = tf.keras.layers.Conv2D(64, 3, 2, padding="same")(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h)))(x)
        concat_list.append(x)
    return tf.keras.layers.concatenate(concat_list)


def tem_block(feature):
    conv_1 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", name="conv_1")(feature)
    gap_1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="gap_1")(conv_1)
    conv_2 = tf.keras.layers.Conv2D(
        256, (1, 1), activation="relu", padding="same", name="conv_2"
    )(gap_1)
    conv_3 = tf.keras.layers.Conv2D(
        256, (1, 1), activation="sigmoid", padding="same", name="conv_3"
    )(conv_2)
    mult_1 = tf.keras.layers.Multiply(name="mult_1")([conv_1, conv_3])
    add_1 = tf.keras.layers.Add(name="add_123")([conv_1, mult_1])

    gap_2 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="gap_2")(add_1)
    cos_sim_1 = tf.keras.layers.Dot(axes=(3), normalize=True, name="cos_sim1")(
        [gap_2, add_1]
    )
    reshape_1 = tf.keras.layers.Flatten()(cos_sim_1)
    transpose_1 = tf.keras.backend.transpose(reshape_1)
    mat_mul_1 = tf.keras.backend.dot(transpose_1, reshape_1)

    conv_4 = tf.keras.layers.Conv2D(256, (1, 1), padding="same", name="conv_4")(add_1)
    permute_1 = tf.keras.layers.Permute((3, 1, 2), name="permute_1")(conv_4)
    reshape_2 = tf.keras.layers.Reshape(
        (-1, int(conv_4.shape[-2]) * int(conv_4.shape[-3])), name="reshape_2"
    )(permute_1)
    mat_mul_2 = tf.keras.backend.dot(reshape_2, mat_mul_1)

    reshape_3 = tf.keras.layers.Reshape(
        (-1, int(conv_4.shape[-2]), int(conv_4.shape[-3])), name="reshape_3"
    )(mat_mul_2)
    permute_2 = tf.keras.layers.Permute((2, 3, 1), name="permute_2")(reshape_3)

    return permute_2


def get_encoder(image_height, image_width):
    input_layer = tf.keras.layers.Input(
        shape=(image_height, image_width, 3), name="input_layer"
    )
    lds_layer = conv_block(input_layer, "conv", 16, (3, 3), strides=(2, 2))
    lds_layer = conv_block(lds_layer, "ds", 32, (3, 3), strides=(2, 2))
    lds_layer = conv_block(lds_layer, "ds", 48, (3, 3), strides=(2, 2))

    gfe_layer = bottleneck_block(lds_layer, 48, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 64, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=1, n=3)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2, 4, 6, 8])

    ff_layer1 = conv_block(
        lds_layer, "conv", 96, (1, 1), padding="same", strides=(1, 1), relu=True
    )

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D(
        (3, 3), strides=(1, 1), depth_multiplier=1, padding="same"
    )(ff_layer2)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(96, 1, 1, padding="same", activation=None)(
        ff_layer2
    )
    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)

    model = tf.keras.Model(inputs=input_layer, outputs=ff_final, name="pcb-encoder")
    return model


def get_decoder(encoder, num_classes):

    inputs = encoder.input
    out = encoder.output
    out = tem_block(out)

    classifier = tf.keras.layers.SeparableConv2D(
        128, (3, 3), padding="same", strides=(1, 1), name="DSConv1_classifier"
    )(out)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = tf.keras.layers.SeparableConv2D(
        128, (3, 3), padding="same", strides=(1, 1), name="DSConv2_classifier"
    )(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
    classifier = tf.keras.layers.Conv2D(num_classes, 1, 1, padding="same")(classifier)
    classifier = tf.keras.activations.softmax(classifier)

    model = tf.keras.models.Model(
        inputs=[inputs], outputs=[classifier], name="pcb-decoder"
    )
    return model


def get_classification(encoder, num_classes):
    inputs = encoder.inputs
    T1 = encoder.output
    gap = tf.keras.layers.GlobalAveragePooling2D()(T1)
    d1 = tf.keras.layers.Dense(128, activation="relu")(gap)
    d2 = tf.keras.layers.Dense(num_classes, activation="softmax")(d1)

    model = tf.keras.Model(inputs=inputs, outputs=d2, name="pcb-classification")
    return model
