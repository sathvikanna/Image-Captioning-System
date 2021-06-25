import pickle
import numpy as np
from PIL import Image
from model import get_feature_model, get_captioning_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def get_image(uploaded_image):
    image = Image.open(uploaded_image)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = image.reshape(1, 224, 224, 3)
    return image


def get_caption(image):
    MAX_LEN = 40
    MAX_WORDS = 50
    tokenizer = get_tokenizer()
    image_to_feature_model = get_feature_model()
    captioning_model = get_captioning_model()
    index_to_word = tokenizer.index_word
    X1 = image_to_feature_model.predict(image).reshape(1, -1)
    current_seed = 'sos'
    generated_caption = ''
    count = 0
    while count < MAX_WORDS:
        count += 1
        encoded = tokenizer.texts_to_sequences([current_seed])
        X2 = np.array(pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN), dtype='float64')
        prediction = np.argmax(captioning_model.predict([X1, X2]))
        sampled = index_to_word[prediction]
        if sampled == 'eos':
            break
        current_seed += ' ' + sampled
        generated_caption += ' ' + sampled 
    caption = generated_caption.strip().capitalize() + " ."
    return caption