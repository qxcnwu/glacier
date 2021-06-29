import tensorflow.keras as keras
import efficientnet.tfkeras as efn

def cnn_model():
    backbone = efn.EfficientNetB4(input_shape=(256, 256, 3), weights=None, include_top=False)
    input = backbone.input
    x = keras.layers.GlobalAveragePooling2D()(backbone.output)
    output = keras.layers.Dense(2, activation=None)(x)
    model = keras.models.Model(inputs=input, outputs=output)
    return model

if __name__ == '__main__':
    model=cnn_model()
    model.summary()