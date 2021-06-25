from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, RepeatVector
from tensorflow.keras.layers import Embedding, LSTM, TimeDistributed
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_feature_model():
    mobilnet = MobileNetV2(include_top=True, weights=None)
    mobilnet_output = mobilnet.layers[-2].output
    model = Model(inputs=[mobilnet.input], outputs=[mobilnet_output])
    model.load_weights("model_weights.h5")
    return model


def get_captioning_model():
    
    EMBED_SIZE = 128
    MAX_LEN = 40
    VOCAB_LEN = 8496

    image_model = Sequential([
        Dense(EMBED_SIZE, input_shape=(1280, ), activation='relu'),
        RepeatVector(MAX_LEN),
    ])

    caption_model = Sequential([
        Embedding(input_dim=VOCAB_LEN+1, output_dim=EMBED_SIZE, input_length=MAX_LEN),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(EMBED_SIZE)),
    ])

    concatenated = Concatenate()([image_model.output, caption_model.output])
    X = LSTM(128, return_sequences=True)(concatenated)
    X = LSTM(512)(X)
    X = Dense(VOCAB_LEN+1)(X)
    output = Activation('softmax')(X)

    captioning_model = Model(inputs=[image_model.input, caption_model.input], outputs=output)
    captioning_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    captioning_model.load_weights("captioning_model_weights.h5")
    return captioning_model