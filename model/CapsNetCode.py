import tensorflow as tf
import tensorflow_io as tfio

#To avoid / 0 error
epsilon = 1e-7   

class CapsuleNetwork(tf.keras.Model):
    def __init__(self, no_of_conv_kernels, no_of_primary_capsules, primary_capsule_vector, no_of_secondary_capsules, secondary_capsule_vector, routing_iterations):
        
        super(CapsuleNetwork, self).__init__()

        # CapsNet parameters
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_capsules = no_of_primary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.secondary_capsule_vector = secondary_capsule_vector
        self.routing_iterations = routing_iterations

        # Initialization
        self.convolution = tf.keras.layers.Conv2D(no_of_conv_kernels, [9, 9], strides=[1, 1], activation='relu', name='ConvolutionLayer')
        self.primary_capsule = tf.keras.layers.Conv2D(no_of_primary_capsules * primary_capsule_vector, [9, 9], strides=[2, 2], padding="same",name="PrimaryCapsule")
        # w = [1, 4608, no_of_secondary_capsules, secondary_capsule_vector, primary_capsule_vector]
        # 4608 = (12 * 12 * 32) where (12*12) is the spatial dimention  and 32 is the dimension of the primary capsule
        self.w = tf.Variable(tf.random.normal([1, 4608, no_of_secondary_capsules,secondary_capsule_vector, primary_capsule_vector]), dtype=tf.float32, name="PoseEstimation", trainable=True)
        
        # Reconstruction Layers
        self.dense_1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.dense_3 = tf.keras.layers.Dense(1024, activation='sigmoid', dtype='float32')
        
    def squash(self, s):
        """
        Applies the squash activation function to vectors
        """
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        return (tf.square(s_norm) / (1 + tf.square(s_norm))) * (s / (s_norm + epsilon))

    def call(self, inputs):
        """
        Forward pass through the Capsule Network
        """
        input_x, y = inputs
        
        # Convolutional and Primary Capsule Layers
        x = self.convolution(input_x)
        x = self.primary_capsule(x)
        
        # Gabor Filter Application and Capsule Formation
        # NOTE : In the original paper, the gabor filter is not applied
        x_gabor = tfio.experimental.filter.gabor(x, freq=0.7, theta=0.9)
        x_gabor = tf.cast(x_gabor, dtype="float32")
        
        u = tf.reshape(x_gabor, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 32))
        u = tf.expand_dims(u, axis=-2)
        u = tf.expand_dims(u, axis=-1)
        u_hat = tf.matmul(self.w, u)
        u_hat = tf.squeeze(u_hat, [4])

        # Dynamic Routing
        b = tf.zeros((x.shape[0], 4608, self.no_of_secondary_capsules, 1))
        for i in range(self.routing_iterations):
            c = tf.nn.softmax(b, axis=-2)
            s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
            v = self.squash(s)
            agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), 
                                             tf.expand_dims(v, axis=-1), transpose_a=True), [4])
            b += agreement

        # Masking
        y = tf.expand_dims(y, axis=-1)
        y = tf.expand_dims(y, axis=1)
        mask = tf.cast(y, dtype=tf.float32)
        v_masked = tf.multiply(mask, v)

        # Reconstruction
        v_reshaped = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector])
        reconstructed_image = self.dense_1(v_reshaped)
        reconstructed_image = self.dropout_1(reconstructed_image)
        reconstructed_image = self.dense_2(reconstructed_image)
        reconstructed_image = self.dropout_2(reconstructed_image)
        reconstructed_image = self.dense_3(reconstructed_image)
        
        return v, reconstructed_image

    def predict_capsule_output(self, inputs):
        """
        Predict the capsule outputs
        """
        x = self.convolution(inputs)
        x = self.primary_capsule(x)
        
        x_gabor = tfio.experimental.filter.gabor(x, freq=0.7, theta=0.9)
        x_gabor = tf.cast(x_gabor, dtype="float32")
        
        u = tf.reshape(x_gabor, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 32))
        u = tf.expand_dims(u, axis=-2)
        u = tf.expand_dims(u, axis=-1)
        u_hat = tf.matmul(self.w, u)
        u_hat = tf.squeeze(u_hat, [4])

        # Dynamic Routing
        b = tf.zeros((x.shape[0], 4608, self.no_of_secondary_capsules, 1))
        for i in range(self.routing_iterations):
            c = tf.nn.softmax(b, axis=-2)
            s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
            v = self.squash(s)
            agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), 
                                             tf.expand_dims(v, axis=-1), transpose_a=True), [4])
            b += agreement
            
        return v

    def regenerate_image(self, inputs):
        """
        Regenerate image from capsule outputs
        """
        v_reshaped = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector])
        reconstructed_image = self.dense_1(v_reshaped)
        reconstructed_image = self.dropout_1(reconstructed_image)
        reconstructed_image = self.dense_2(reconstructed_image)
        reconstructed_image = self.dropout_2(reconstructed_image)
        reconstructed_image = self.dense_3(reconstructed_image)
        return reconstructed_image
