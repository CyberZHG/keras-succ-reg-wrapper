# Keras Successive Regularization Wrapper

[![Travis](https://travis-ci.org/CyberZHG/keras-succ-reg-wrapper.svg)](https://travis-ci.org/CyberZHG/keras-succ-reg-wrapper)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-succ-reg-wrapper/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-succ-reg-wrapper)

A wrapper that slows down the updates of trainable weights.

![](https://user-images.githubusercontent.com/853842/50722430-dce6c580-1109-11e9-834d-7dc92b9221db.png)

## Install

```bash
pip install keras-succ-reg-wrapper
```

## Usage

```python
import keras
from keras_succ_reg_wrapper import SuccReg

input_layer = keras.layers.Input(shape=(1,), name='Input')
dense_layer = SuccReg(
    layer=keras.layers.Dense(units=1, name='Dense'),
    regularizer=keras.regularizers.L1L2(l2=1e-3),  # Any regularizer
    name='Output',
)(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
model.compile(optimizer='adam', loss='mse')
model.summary()
```
