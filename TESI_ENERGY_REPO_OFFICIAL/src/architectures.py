import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Bidirectional, Attention, GlobalAveragePooling1D

def build_lstm(n_in, n_features, n_out=1):
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_in, n_features)))
    model.add(Dense(n_out))
    model.compile(loss='mae', optimizer='adam')
    return model

def build_bilstm(n_in, n_features, n_out=1):
    model = Sequential()
    # Come da Tesi Cap 4.1.2: LSTM bidirezionale a 70 unità (70 forward + 70 backward = 140)
    model.add(Bidirectional(LSTM(70), input_shape=(n_in, n_features)))
    model.add(Dense(n_out))
    model.compile(loss='mae', optimizer='adam')
    return model

def build_cnn_lstm(n_in, n_features, n_out=1):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_in, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(n_out))
    model.compile(loss='mae', optimizer='adam')
    return model

def build_cnn_bilstm(n_in, n_features, n_out=1):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_in, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    # Come da Tesi Cap 4.1.2: Rete a valle composta da 70 neuroni (140 bi-direzionali)
    model.add(Bidirectional(LSTM(70)))
    model.add(Dense(n_out))
    model.compile(loss='mae', optimizer='adam')
    return model

def build_cnn_bilstm_attention(n_in, n_features, n_out=1):
    inputs = Input(shape=(n_in, n_features))
    
    # CNN Layer
    cnn_out = Conv1D(filters=256, kernel_size=3, activation='relu')(inputs)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    
    # Attention Layer sulla CNN prima della LSTM (come da versione Notebook 11)
    attn_out = Attention()([cnn_out, cnn_out])
    
    # Bi-LSTM Layer
    lstm_out = Bidirectional(LSTM(70))(attn_out)
    
    # Output Layer
    outputs = Dense(n_out)(lstm_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mae', optimizer='adam')
    return model
