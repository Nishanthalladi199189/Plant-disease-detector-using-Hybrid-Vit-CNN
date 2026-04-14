"""Plant Disease Detection App"""
import os, json, numpy as np, streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@keras.utils.register_keras_serializable()
class CustomCNNBlock(keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=3, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size, self.dropout_rate = filters, kernel_size, dropout_rate
    def build(self, input_shape):
        self.conv1 = keras.layers.Conv2D(self.filters, self.kernel_size, padding='same', activation='relu')
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(self.filters, self.kernel_size, padding='same', activation='relu')
        self.bn2 = keras.layers.BatchNormalization()
        self.pool = keras.layers.MaxPooling2D(pool_size=2)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool(x)
        return self.dropout(x, training=training)
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'dropout_rate': self.dropout_rate})
        return config

@keras.utils.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
    def build(self, input_shape):
        self.att = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim//self.num_heads)
        self.ffn = keras.Sequential([keras.layers.Dense(self.ff_dim, activation='relu'), keras.layers.Dense(self.embed_dim)])
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    def call(self, inputs, training=None):
        attn = self.att(inputs, inputs, training=training)
        attn = self.dropout1(attn, training=training)
        out1 = self.ln1(inputs + attn)
        ffn = self.ffn(out1)
        ffn = self.dropout2(ffn, training=training)
        return self.ln2(out1 + ffn)
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'dropout_rate': self.dropout_rate})
        return config

try:
    with open('class_indices.json', 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
        CLASS_NAMES = [class_indices[str(i)] for i in range(38)]
except Exception as e:
    st.error(f"Error loading class_indices.json: {e}")
    # Fallback class names
    CLASS_NAMES = [f"Class_{i}" for i in range(38)]

MODEL_PATH = 'trained_model/hybrid_vit_cnn_plant_disease_model.keras'
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model():
    try:
        custom_objects = {'CustomCNNBlock': CustomCNNBlock, 'TransformerBlock': TransformerBlock}
        return keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None

def preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    return np.expand_dims(np.array(image) / 255.0, axis=0)

def main():
    st.set_page_config(page_title='Plant Disease Detection', page_icon=':seedling:')
    st.title(':seedling: Plant Disease Detection')
    st.markdown('Upload a plant leaf image to detect diseases')
    model = load_model()
    if model is None:
        st.error('Failed to load model!')
        return
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        if st.button('Analyze'):
            with st.spinner('Analyzing...'):
                processed = preprocess_image(image)
                predictions = model.predict(processed, verbose=0)
                top_idx = np.argsort(predictions[0])[-3:][::-1]
                st.success('Analysis Complete!')
                st.markdown('### Results')
                for i, idx in enumerate(top_idx):
                    disease, conf = CLASS_NAMES[idx], predictions[0][idx] * 100
                    st.markdown(f'{i+1}. **{disease}** - {conf:.2f}%' if i==0 else f'{i+1}. {disease} - {conf:.2f}%')

if __name__ == '__main__':
    main()
