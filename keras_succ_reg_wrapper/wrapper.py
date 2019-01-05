import keras
import keras.backend as K


class SuccReg(keras.layers.Wrapper):

    def __init__(self, layer, regularizer, **kwargs):
        super(SuccReg, self).__init__(layer, **kwargs)
        self.supports_masking = True
        self.regularizer = keras.regularizers.get(regularizer)
        self.iteration = None
        self.prev_weights = None

    def get_config(self):
        config = {
            'regularizer': self.regularizer,
        }
        base_config = super(SuccReg, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape=None):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.iteration = self.add_weight(
            shape=(),
            trainable=False,
            initializer='zeros',
            name='iter',
        )
        self.prev_weights = []
        for i, weight in enumerate(self.layer.trainable_weights):
            self.prev_weights.append(self.add_weight(
                shape=weight.shape,
                trainable=False,
                initializer='zeros',
                name='W_%d' % i,
            ))
            self.add_loss(K.switch(
                K.equal(self.iteration, K.constant(0.0)),
                0.0,
                self.regularizer(weight - self.prev_weights[-1]),
            ))
        super(SuccReg, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def call(self, inputs, **kwargs):
        updates = [K.update_add(self.iteration, 1.0)]
        for i, weight in enumerate(self.layer.trainable_weights):
            updates.append(K.update(self.prev_weights[i], weight))
        self.add_update(updates, inputs)
        return self.layer.call(inputs, **kwargs)

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights + self.layer.non_trainable_weights
