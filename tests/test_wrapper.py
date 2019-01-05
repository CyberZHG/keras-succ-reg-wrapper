import os
import tempfile
import random
from unittest import TestCase
import keras
import numpy as np
from keras_succ_reg_wrapper import SuccReg


class TestWrapper(TestCase):

    @staticmethod
    def _save_and_load(model):
        model_path = os.path.join(tempfile.gettempdir(), 'keras_succ_reg_wrapper_%f.h5' % random.random())
        model.save(model_path)
        return keras.models.load_model(model_path, custom_objects={'SuccReg': SuccReg})

    @staticmethod
    def _build_dense_model(regularizer):
        input_layer = keras.layers.Input(shape=(1,), name='Input')
        dense_layer = SuccReg(
            layer=keras.layers.Dense(units=1, name='Dense'),
            regularizer=regularizer,
            name='Output',
        )(input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    def test_fit_rate(self):
        model = self._build_dense_model(keras.regularizers.L1L2(l2=1e10))
        model = self._save_and_load(model)
        model.fit(x=np.ones((10000, 1)), y=np.ones((10000, 1)) * 0.6, epochs=10)
        model = self._save_and_load(model)
        diff_large = abs(model.predict(np.ones((1, 1)))[0, 0].tolist() - 0.6)

        model = self._build_dense_model(keras.regularizers.L1L2(l2=1e-3))
        model = self._save_and_load(model)
        model.fit(x=np.ones((10000, 1)), y=np.ones((10000, 1)) * 0.6, epochs=10)
        model = self._save_and_load(model)
        diff_small = abs(model.predict(np.ones((1, 1)))[0, 0].tolist() - 0.6)

        self.assertGreater(diff_large, diff_small * 100, (diff_large, diff_small))
