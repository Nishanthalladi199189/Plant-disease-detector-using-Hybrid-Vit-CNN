"""
Plant Disease Detection App - Simple Version
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================
# CUSTOM LAYERS (Required for loading the hybrid model)
# ============================================

@keras.utils.register_keras_serializable()
class CustomCNNBlock(keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=3, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
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
class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
    def build(self, input_shape):
        self.query_dense = keras.layers.Dense(self.embed_dim)
        self.key_dense = keras.layers.Dense(self.embed_dim)
        self.value_dense = keras.layers.Dense(self.embed_dim)
        self.combine_heads = keras.layers.Dense(self.embed_dim)
        super().build(input_shape)
    
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads})
        return config

@keras.utils.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, ff_dim=3072, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.ffn = keras.Sequential([
            keras.layers.Dense(self.ff_dim, activation="relu"),
            keras.layers.Dense(self.embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'dropout_rate': self.dropout_rate})
        return config

# Load class names
try:
    with open('class_indices.json', 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
        CLASS_NAMES = [class_indices[str(i)] for i in range(38)]
except:
    # Fallback class names
    CLASS_NAMES = [f"Class_{i}" for i in range(38)]

MODEL_PATH = "trained_model/hybrid_vit_cnn_plant_disease_model.keras"
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model():
    """Load model with custom objects"""
    try:
        custom_objects = {
            'CustomCNNBlock': CustomCNNBlock,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock
        }
        return keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model"""
    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def main():
    st.set_page_config(page_title="Plant Disease Detection", page_icon=":seedling:")
    
    st.title(":seedling: Plant Disease Detection")
    st.markdown("Upload a plant leaf image to detect diseases")
    
    model = load_model()
    
    if model is None:
        st.error("Failed to load model!")
        return
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                try:
                    processed = preprocess_image(image)
                    predictions = model.predict(processed, verbose=0)
                    
                    # Get top 3 predictions
                    top_idx = np.argsort(predictions[0])[-3:][::-1]
                    
                    st.success("Analysis Complete!")
                    st.markdown("### Results")
                    
                    for i, idx in enumerate(top_idx):
                        disease = CLASS_NAMES[idx]
                        conf = predictions[0][idx] * 100
                        if i == 0:
                            st.markdown(f"{i+1}. **{disease}** - {conf:.2f}%")
                        else:
                            st.markdown(f"{i+1}. {disease} - {conf:.2f}%")
                
                except Exception as e:
                    st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
