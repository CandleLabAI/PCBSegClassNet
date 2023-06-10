# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Preprocessing super resolution
"""

from tensorflow.keras import layers
import tensorflow as tf

class EDSRModel(tf.keras.Model):
    """
    Main class of super resolution model
    """
    def train_step(self, data):
        """
        forward pass function
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inp, target = data

        with tf.GradientTape() as tape:
            y_pred = self(inp, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(target, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(target, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, inputs):
        """
        prediction function
        """
        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        out = tf.cast(tf.expand_dims(inputs, axis=0), tf.float32)
        # Passing low resolution image to model
        super_resolution_img = self(out, training=False)
        # Clips the tensor from min(0) to max(255)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        # Rounds the values of a tensor to the nearest integer
        super_resolution_img = tf.round(super_resolution_img)
        # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
        super_resolution_img = tf.squeeze(
            tf.cast(super_resolution_img, tf.uint8), axis=0
        )
        return super_resolution_img

# Residual Block
def resblock(inputs):
    """
    residual convolution block
    """
    out = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    out = layers.Conv2D(64, 3, padding="same")(out)
    out = layers.Add()([inputs, out])
    return out

# Upsampling Block
def upsampling(inputs, factor=2, **kwargs):
    """
    upsampling convolution block
    """
    out = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    out = tf.nn.depth_to_space(out, block_size=factor)
    out = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(out)
    out = tf.nn.depth_to_space(out, block_size=factor)
    return out

def get_edsr_model(num_filters, num_of_residual_blocks):
    """
    Get the final model from the given arguments
    """
    # Flexible Inputs to input_layer
    input_layer = layers.Input(shape=(None, None, 3))
    # Scaling Pixel Values
    out = layers.Rescaling(scale=1.0 / 255)(input_layer)
    out = x_new = layers.Conv2D(num_filters, 3, padding="same")(out)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = resblock(x_new)

    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    out = layers.Add()([out, x_new])

    out = upsampling(out)
    out = layers.Conv2D(3, 3, padding="same")(out)

    output_layer = layers.Rescaling(scale=255)(out)
    return EDSRModel(input_layer, output_layer)
