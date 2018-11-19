from keras import backend as K
from keras.engine.base_layer import Layer, InputSpec

class UniformNoise(Layer):
    minval: float
    maxval: float

    def __init__(self, minval, maxval, **kwargs):
        super(UniformNoise, self).__init__(**kwargs)
        self.minval = minval
        self.maxval = maxval
        self.supports_masking = True

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_uniform(shape=K.shape(inputs),minval=self.minval,maxval=self.maxval)
        return K.in_train_phase(noised, noised, training=training)

    def get_config(self):
        config = {'minval':self.minval, 'maxval':self.maxval  }
        base_config = super(UniformNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape