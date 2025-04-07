from keras import layers
from keras.models import Model
from utils import build_residual_block

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")


    normalized_inputs = layers.Lambda(lambda x: x / 255)(inputs)


    x1 = build_residual_block(normalized_inputs, 16, activation_type=activation, use_skip_conv=True, stride_value=1, dropout_rate=dropout)
    x2 = build_residual_block(x1, 16, activation_type=activation, use_skip_conv=True, stride_value=2, dropout_rate=dropout)
    x3 = build_residual_block(x2, 16, activation_type=activation, use_skip_conv=False, stride_value=1, dropout_rate=dropout)
    x4 = build_residual_block(x3, 32, activation_type=activation, use_skip_conv=True, stride_value=2, dropout_rate=dropout)
    x5 = build_residual_block(x4, 32, activation_type=activation, use_skip_conv=False, stride_value=1, dropout_rate=dropout)
    x6 = build_residual_block(x5, 64, activation_type=activation, use_skip_conv=True, stride_value=2, dropout_rate=dropout)
    x7 = build_residual_block(x6, 64, activation_type=activation, use_skip_conv=True, stride_value=1, dropout_rate=dropout)
    x8 = build_residual_block(x7, 64, activation_type=activation, use_skip_conv=False, stride_value=1, dropout_rate=dropout)
    x9 = build_residual_block(x8, 64, activation_type=activation, use_skip_conv=False, stride_value=1, dropout_rate=dropout)


    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)
    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)


    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)


    model = Model(inputs=inputs, outputs=output)
    return model
