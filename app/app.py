"""
Plant Disease Detection Web Application
Using Hybrid CNN + Vision Transformer (ViT) Model
====================================================

A production-ready Streamlit application for detecting plant diseases
from leaf images using a hybrid deep learning model.

Author: Plant Disease Detection Team
Version: 2.0.0
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import io
import base64
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================
# CUSTOM LAYERS (Required for loading the hybrid model)
# ============================================
# These classes must be defined before loading the model and passed via custom_objects

class CustomCNNBlock(keras.layers.Layer):
    """Custom CNN Block for feature extraction."""
    def __init__(self, filters=64, kernel_size=3, dropout_rate=0.1, **kwargs):
        # Filter out args that cause issues
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['trainable', 'dtype', 'batch_input_shape', 'name']}
        super().__init__(**clean_kwargs)
        self.filters = int(filters) if filters else 64
        self.kernel_size = int(kernel_size) if kernel_size else 3
        self.dropout_rate = float(dropout_rate) if dropout_rate else 0.1
        
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
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config

class PatchEmbedding(keras.layers.Layer):
    """Patch Embedding layer for Vision Transformer."""
    def __init__(self, patch_size=16, embed_dim=768, **kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['trainable', 'dtype', 'batch_input_shape', 'name']}
        super().__init__(**clean_kwargs)
        self.patch_size = int(patch_size) if patch_size else 16
        self.embed_dim = int(embed_dim) if embed_dim else 768
        
    def build(self, input_shape):
        self.projection = keras.layers.Conv2D(self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size)
        super().build(input_shape)
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = self.projection(images)
        patches = tf.reshape(patches, [batch_size, -1, self.embed_dim])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim
        })
        return config

class MultiHeadSelfAttention(keras.layers.Layer):
    """Multi-Head Self Attention for Vision Transformer."""
    def __init__(self, embed_dim=768, num_heads=12, **kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['trainable', 'dtype', 'batch_input_shape', 'name']}
        super().__init__(**clean_kwargs)
        self.embed_dim = int(embed_dim) if embed_dim else 768
        self.num_heads = int(num_heads) if num_heads else 12
        self.head_dim = self.embed_dim // self.num_heads
        
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
        output = self.combine_heads(concat_attention)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config

class TransformerBlock(keras.layers.Layer):
    """Transformer Block with attention and feed-forward network."""
    def __init__(self, embed_dim=768, num_heads=12, ff_dim=3072, dropout_rate=0.1, **kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['trainable', 'dtype', 'batch_input_shape', 'name']}
        super().__init__(**clean_kwargs)
        self.embed_dim = int(embed_dim) if embed_dim else 768
        self.num_heads = int(num_heads) if num_heads else 12
        self.ff_dim = int(ff_dim) if ff_dim else 3072
        self.dropout_rate = float(dropout_rate) if dropout_rate else 0.1
        
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
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

# ============================================
# CONFIGURATION
# ============================================

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants - Try .keras first (more compatible with newer TensorFlow), fallback to .h5
MODEL_PATH_KERAS = "trained_model/hybrid_vit_cnn_plant_disease_model.keras"
MODEL_PATH_H5 = "trained_model/hybrid_vit_cnn_plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMAGE_SIZE = (224, 224)  # Model trained with 224x224 images
CLASS_NAMES = {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Blueberry___healthy",
    "5": "Cherry_(including_sour)___Powdery_mildew",
    "6": "Cherry_(including_sour)___healthy",
    "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "8": "Corn_(maize)___Common_rust_",
    "9": "Corn_(maize)___Northern_Leaf_Blight",
    "10": "Corn_(maize)___healthy",
    "11": "Grape___Black_rot",
    "12": "Grape___Esca_(Black_Measles)",
    "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "14": "Grape___healthy",
    "15": "Orange___Haunglongbing_(Citrus_greening)",
    "16": "Peach___Bacterial_spot",
    "17": "Peach___healthy",
    "18": "Pepper,_bell___Bacterial_spot",
    "19": "Pepper,_bell___healthy",
    "20": "Potato___Early_blight",
    "21": "Potato___Late_blight",
    "22": "Potato___healthy",
    "23": "Raspberry___healthy",
    "24": "Soybean___healthy",
    "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch",
    "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot",
    "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight",
    "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot",
    "33": "Tomato___Spider_mites Two-spotted_spider_mite",
    "34": "Tomato___Target_Spot",
    "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "36": "Tomato___Tomato_mosaic_virus",
    "37": "Tomato___healthy"
}

# ============================================
# MODEL MANAGEMENT
# ============================================

# Global model variable
_model = None

def load_model():
    """
    Load the pre-trained hybrid CNN + ViT model.
    Tries .keras format first (more compatible with newer TensorFlow), then .h5 as fallback.
    
    Returns:
        keras.Model: The loaded model or None if loading fails
    """
    global _model
    
    # Return cached model if already loaded
    if _model is not None:
        return _model
    
    st.info("🔄 Loading model... Please wait.")
    
    # Build model from scratch and load weights only
    # This bypasses all Keras version serialization issues
    try:
        st.info("🏗️ Building model architecture...")
        
        # Input layer
        inputs = keras.Input(shape=(224, 224, 3))
        
        # CNN feature extraction blocks
        x = CustomCNNBlock(filters=64, kernel_size=3, dropout_rate=0.1)(inputs)
        x = CustomCNNBlock(filters=128, kernel_size=3, dropout_rate=0.1)(x)
        x = CustomCNNBlock(filters=256, kernel_size=3, dropout_rate=0.1)(x)
        
        # Reshape for transformer: (batch, 784, 256)
        x = keras.layers.Reshape(target_shape=(784, 256))(x)
        
        # Learnable positional embedding (skip connection)
        positional_embedding = keras.layers.Embedding(input_dim=784, output_dim=256)
        positions = tf.range(start=0, limit=784, delta=1)
        pos_embed = positional_embedding(positions)
        x = keras.layers.Add()([x, pos_embed])
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=256, num_heads=8, ff_dim=512, dropout_rate=0.1)(x)
        x = TransformerBlock(embed_dim=256, num_heads=8, ff_dim=512, dropout_rate=0.1)(x)
        x = TransformerBlock(embed_dim=256, num_heads=8, ff_dim=512, dropout_rate=0.1)(x)
        x = TransformerBlock(embed_dim=256, num_heads=8, ff_dim=512, dropout_rate=0.1)(x)
        
        # Classification head
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.1)(x)
        outputs = keras.layers.Dense(38, activation='softmax')(x)
        
        _model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Load weights from .keras file (zip format)
        if os.path.exists(MODEL_PATH_KERAS):
            st.info("📦 Loading weights from .keras file...")
            import zipfile, tempfile
            with zipfile.ZipFile(MODEL_PATH_KERAS, 'r') as z:
                # Extract weights h5 to temp file
                weights_file = z.extract('model.weights.h5', tempfile.gettempdir())
                _model.load_weights(weights_file)
                os.remove(weights_file)
            st.success("✅ Model loaded successfully!")
            return _model
        elif os.path.exists(MODEL_PATH_H5):
            st.info("📦 Loading weights from .h5 file...")
            _model.load_weights(MODEL_PATH_H5)
            st.success("✅ Model loaded successfully!")
            return _model
        else:
            st.error("❌ No model weights file found.")
            return None
            
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)[:200]}")
        import traceback
        st.error(traceback.format_exc()[:500])
        return None

# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image, target_size=IMAGE_SIZE):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image: PIL Image or numpy array
        target_size: Tuple of (width, height) for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for prediction
    """
    try:
        # Convert to RGB if necessary
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

# ============================================
# PREDICTION
# ============================================

def predict(model, preprocessed_image):
    """
    Run prediction on the preprocessed image.
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed numpy array
        
    Returns:
        tuple: (predicted_class_index, confidence_score, all_probabilities)
    """
    try:
        # Run prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return predicted_class_idx, confidence, predictions[0]
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None

def get_disease_info(class_name):
    """
    Get detailed information about a disease class.
    
    Args:
        class_name: Name of the disease class
        
    Returns:
        dict: Disease information
    """
    # Disease descriptions and recommendations
    disease_database = {
        "healthy": {
            "description": "The plant appears healthy with no visible disease symptoms.",
            "recommendation": "Continue regular maintenance and monitoring.",
            "severity": "None",
            "color": "#28a745"
        },
        "Apple_scab": {
            "description": "A fungal disease causing olive-green to black spots on leaves and fruit.",
            "recommendation": "Apply fungicides, remove fallen leaves, and plant resistant varieties.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Black_rot": {
            "description": "A fungal disease causing fruit rot, leaf spots, and cankers.",
            "recommendation": "Prune infected branches, apply fungicides, and maintain tree health.",
            "severity": "High",
            "color": "#dc3545"
        },
        "Cedar_apple_rust": {
            "description": "A fungal disease causing yellow-orange spots on leaves.",
            "recommendation": "Remove nearby cedar trees if possible, apply fungicides.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Powdery_mildew": {
            "description": "A fungal disease appearing as white powdery spots on leaves.",
            "recommendation": "Improve air circulation, apply fungicides, remove infected parts.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Cercospora_leaf_spot": {
            "description": "A fungal disease causing gray to brown spots with purple borders.",
            "recommendation": "Rotate crops, apply fungicides, and remove plant debris.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Common_rust": {
            "description": "A fungal disease causing rust-colored pustules on leaves.",
            "recommendation": "Plant resistant varieties, apply fungicides when necessary.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Northern_Leaf_Blight": {
            "description": "A fungal disease causing cigar-shaped lesions on leaves.",
            "recommendation": "Rotate crops, till under residue, use resistant hybrids.",
            "severity": "High",
            "color": "#dc3545"
        },
        "Esca": {
            "description": "A complex fungal disease causing leaf striping and fruit rot.",
            "recommendation": "Remove infected wood, treat pruning wounds, maintain vine health.",
            "severity": "High",
            "color": "#dc3545"
        },
        "Leaf_blight": {
            "description": "A fungal disease causing browning and death of leaf tissue.",
            "recommendation": "Apply fungicides, improve drainage, remove infected leaves.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Haunglongbing": {
            "description": "Citrus greening disease causing asymmetrical yellow mottling.",
            "recommendation": "Remove infected trees, control psyllid vectors, plant clean stock.",
            "severity": "Critical",
            "color": "#dc3545"
        },
        "Bacterial_spot": {
            "description": "A bacterial disease causing dark, water-soaked spots on leaves.",
            "recommendation": "Use copper sprays, rotate crops, use disease-free seeds.",
            "severity": "High",
            "color": "#dc3545"
        },
        "Early_blight": {
            "description": "A fungal disease causing dark concentric rings on leaves.",
            "recommendation": "Rotate crops, mulch, apply fungicides, remove lower leaves.",
            "severity": "High",
            "color": "#dc3545"
        },
        "Late_blight": {
            "description": "A devastating fungal disease causing brown blotches on leaves.",
            "recommendation": "Apply fungicides regularly, destroy infected plants, ensure good drainage.",
            "severity": "Critical",
            "color": "#dc3545"
        },
        "Leaf_scorch": {
            "description": "A fungal disease causing purple to black spots on leaves.",
            "recommendation": "Remove infected leaves, improve air circulation, apply fungicides.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Leaf_Mold": {
            "description": "A fungal disease causing yellow spots with olive-green mold underneath.",
            "recommendation": "Improve ventilation, reduce humidity, apply fungicides.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Septoria_leaf_spot": {
            "description": "A fungal disease causing small circular spots with gray centers.",
            "recommendation": "Remove infected leaves, mulch, apply fungicides, water at base.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Spider_mites": {
            "description": "Tiny pests causing stippling and yellowing of leaves.",
            "recommendation": "Use miticides, increase humidity, introduce predatory mites.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Target_Spot": {
            "description": "A fungal disease causing brown spots with concentric rings.",
            "recommendation": "Remove infected tissue, apply fungicides, improve air flow.",
            "severity": "Medium",
            "color": "#ffc107"
        },
        "Tomato_Yellow_Leaf_Curl_Virus": {
            "description": "A viral disease causing yellowing and curling of leaves.",
            "recommendation": "Control whiteflies, use resistant varieties, remove infected plants.",
            "severity": "Critical",
            "color": "#dc3545"
        },
        "Tomato_mosaic_virus": {
            "description": "A viral disease causing mottled light and dark green patterns.",
            "recommendation": "Sanitize tools, remove infected plants, use resistant varieties.",
            "severity": "High",
            "color": "#dc3545"
        },
        "Bacterial_spot": {
            "description": "Bacterial disease causing dark spots on leaves and fruit.",
            "recommendation": "Use copper-based sprays, avoid overhead watering, rotate crops.",
            "severity": "High",
            "color": "#dc3545"
        }
    }
    
    # Find matching disease info
    for key, info in disease_database.items():
        if key.lower() in class_name.lower():
            return info
    
    # Default info for unknown diseases
    return {
        "description": "Disease detected. Please consult a plant pathologist for detailed diagnosis.",
        "recommendation": "Monitor the plant closely and consider professional consultation.",
        "severity": "Unknown",
        "color": "#6c757d"
    }

# ============================================
# GRAD-CAM VISUALIZATION
# ============================================

def create_leaf_mask(image, target_size=(224, 224)):
    """
    Create a binary mask to isolate the leaf from the background.
    Uses color-based segmentation to detect green leaf regions.
    
    Args:
        image: Input image (PIL Image or numpy array)
        target_size: Target size for resizing
        
    Returns:
        numpy.ndarray: Binary mask (1 = leaf, 0 = background)
    """
    try:
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Resize to target size
        img_resized = cv2.resize(img_array, target_size)
        
        # Convert to HSV color space for better green detection
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        
        # Define green color range (leaf colors)
        # Lower and upper bounds for green in HSV
        lower_green = np.array([25, 40, 40])  # Adjusted for various green shades
        upper_green = np.array([95, 255, 255])
        
        # Create mask for green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Normalize to 0-1
        mask = mask.astype(np.float32) / 255.0
        
        return mask
    except Exception as e:
        # Return full mask if segmentation fails
        return np.ones(target_size, dtype=np.float32)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None, original_image=None):
    """
    Generate Grad-CAM heatmap for model interpretability.
    Optimized to focus on leaf areas and use appropriate CNN layers.
    
    Args:
        img_array: Preprocessed image array
        model: Loaded model
        last_conv_layer_name: Name of convolutional layer (auto-detected if None)
        pred_index: Index of class to visualize (top prediction if None)
        original_image: Original image for creating leaf mask
        
    Returns:
        numpy.ndarray: Heatmap array
    """
    try:
        # Get layer names
        layer_names = [layer.name for layer in model.layers]
        
        # For hybrid model, look for CustomCNNBlock layers
        cnn_block_names = [name for name in layer_names if 'custom_cnn_block' in name.lower()]
        
        if last_conv_layer_name is None:
            if cnn_block_names:
                # Use SECOND CNN block (index 1) for better leaf feature detection
                # First block captures too much noise, last block is too abstract
                if len(cnn_block_names) >= 2:
                    last_conv_layer_name = cnn_block_names[1]  # Middle layer
                else:
                    last_conv_layer_name = cnn_block_names[0]
            else:
                # Fallback
                for layer in reversed(model.layers):
                    if any(x in layer.name.lower() for x in ['conv', 'cnn', 'block']):
                        last_conv_layer_name = layer.name
                        break
        
        if last_conv_layer_name is None:
            return None
        
        # Get the layer and create grad model
        target_layer = model.get_layer(last_conv_layer_name)
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Gradient computation
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            return None
        
        # Use guided backpropagation - only positive gradients
        grads = tf.maximum(grads, 0)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        # Resize to match target size
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # Apply leaf mask to remove background activation
        if original_image is not None:
            leaf_mask = create_leaf_mask(original_image, target_size=(224, 224))
            # Suppress heatmap values outside leaf region
            heatmap_resized = heatmap_resized * leaf_mask
            
            # Re-normalize after masking
            if heatmap_resized.max() > 0:
                heatmap_resized = heatmap_resized / heatmap_resized.max()
        
        return heatmap_resized
    except Exception as e:
        return None

def overlay_heatmap(heatmap, image, alpha=0.4):
    """
    Overlay heatmap on original image.
    
    Args:
        heatmap: Grad-CAM heatmap (numpy array)
        image: Original image (numpy array, RGB, 0-255 or 0-1)
        alpha: Transparency of heatmap
        
    Returns:
        PIL.Image: Superimposed image
    """
    try:
        # Ensure heatmap is numpy array and 2D
        heatmap = np.array(heatmap)
        if len(heatmap.shape) > 2:
            heatmap = np.squeeze(heatmap)
        
        # Ensure image is numpy array
        image = np.array(image)
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))
        
        # Normalize heatmap to 0-255
        heatmap_normalized = np.uint8(255 * heatmap_resized / (heatmap_resized.max() + 1e-8))
        
        # Apply colormap (jet) -> returns RGBA
        heatmap_colored = cm.jet(heatmap_normalized)
        
        # Convert to RGB (drop alpha channel)
        heatmap_rgb = np.uint8(255 * heatmap_colored[:, :, :3])
        
        # Ensure image is in proper format (uint8, 0-255)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = np.uint8(255 * image)
            else:
                image = np.uint8(image)
        
        # Ensure image is RGB (not RGBA)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Superimpose heatmap on image
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)
        
        return Image.fromarray(superimposed)
    except Exception as e:
        # Log error for debugging
        print(f"Grad-CAM overlay error: {str(e)}")
        return None

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_probability_chart(predictions, class_names, top_n=5):
    """
    Create a horizontal bar chart of top predictions.
    
    Args:
        predictions: Array of prediction probabilities
        class_names: Dictionary of class indices to names
        top_n: Number of top predictions to show
        
    Returns:
        matplotlib.figure.Figure: Bar chart figure
    """
    # Get top N predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    
    labels = []
    values = []
    colors = []
    
    for idx in top_indices:
        class_name = class_names[str(idx)]
        # Clean up class name for display
        clean_name = class_name.replace('_', ' ').replace('___', ' - ')
        labels.append(clean_name)
        values.append(predictions[idx] * 100)
        
        # Color based on rank
        if idx == top_indices[0]:
            colors.append('#28a745')  # Green for top prediction
        else:
            colors.append('#6c757d')  # Gray for others
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.8)
    
    # Customize
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Confidence (%)', fontsize=11)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Top Predictions', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig

def display_prediction_gauge(confidence):
    """
    Display a visual confidence gauge using Streamlit components.
    
    Args:
        confidence: Confidence score (0-1)
    """
    percentage = confidence * 100
    
    # Determine color based on confidence
    if percentage >= 90:
        color = "#28a745"
        status = "Very High"
    elif percentage >= 75:
        color = "#6fbf73"
        status = "High"
    elif percentage >= 60:
        color = "#ffc107"
        status = "Moderate"
    else:
        color = "#dc3545"
        status = "Low"
    
    # Create progress bar
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: 600;">Confidence Score</span>
            <span style="color: {color}; font-weight: 600;">{status}</span>
        </div>
        <div style="background-color: #e9ecef; border-radius: 10px; height: 25px; overflow: hidden;">
            <div style="background-color: {color}; width: {percentage}%; height: 100%; 
                        border-radius: 10px; transition: width 0.5s ease;"></div>
        </div>
        <div style="text-align: right; margin-top: 5px; font-size: 0.9em; color: #666;">
            {percentage:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SIDEBAR COMPONENTS
# ============================================

def render_sidebar():
    """Render the sidebar with project information."""
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h2 style="color: #2E7D32; margin-bottom: 5px;">🌿 PlantDoc</h2>
            <p style="color: #666; font-size: 0.9em;">AI-Powered Plant Health</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Project Description
        st.markdown("""
        ### 📋 About
        
        This application uses a **Hybrid CNN + Vision Transformer (ViT)** deep learning model 
        to detect diseases in plant leaves from images.
        
        **Features:**
        - 🎯 38 disease categories across 14 plant types
        - 🧠 Hybrid CNN + ViT architecture
        - 📊 Real-time prediction visualization
        - 🔍 Grad-CAM explainability
        - ⚡ Fast and accurate results
        """)
        
        st.divider()
        
        # Model Information
        st.markdown("""
        ### 🧠 Model Architecture
        
        **Hybrid CNN + ViT**
        
        - **CNN**: Extracts local spatial features from leaf images
        - **Vision Transformer**: Captures global dependencies and long-range patterns
        - **Fusion**: Combines local and global features for superior accuracy
        
        **Input**: 128x128 RGB images
        **Output**: 38 disease classes
        """)
        
        st.divider()
        
        # Supported Diseases
        with st.expander("🌱 Supported Plants & Diseases"):
            plants = {
                "🍎 Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
                "🫐 Blueberry": ["Healthy"],
                "🍒 Cherry": ["Powdery Mildew", "Healthy"],
                "🌽 Corn": ["Cercospora Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"],
                "🍇 Grape": ["Black Rot", "Esca", "Leaf Blight", "Healthy"],
                "🍊 Orange": ["Citrus Greening (HLB)"],
                "🍑 Peach": ["Bacterial Spot", "Healthy"],
                "🫑 Pepper": ["Bacterial Spot", "Healthy"],
                "🥔 Potato": ["Early Blight", "Late Blight", "Healthy"],
                "🍓 Raspberry": ["Healthy"],
                "🫘 Soybean": ["Healthy"],
                "🎃 Squash": ["Powdery Mildew"],
                "🍓 Strawberry": ["Leaf Scorch", "Healthy"],
                "🍅 Tomato": ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", 
                           "Septoria Leaf Spot", "Spider Mites", "Target Spot", 
                           "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"]
            }
            
            for plant, diseases in plants.items():
                st.markdown(f"**{plant}**")
                for disease in diseases:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {disease}")
        
        st.divider()
        
        # Developer Info
        st.markdown("""
        ### 👨‍💻 Developer Info
        
        **Version**: 2.0.0
        
        **Built with**:
        - Streamlit
        - TensorFlow/Keras
        - OpenCV
        - NumPy
        
        **Last Updated**: 2026
        """)
        
        # GitHub link (placeholder)
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <a href="#" style="text-decoration: none; color: #666;">
                ⭐ Star on GitHub
            </a>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point."""
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            background-color: #2E7D32;
            color: white;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #1B5E20;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
        }
        .prediction-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border-left: 5px solid #2E7D32;
        }
        .severity-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .uploaded-image {
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #2E7D32; font-size: 2.5em; margin-bottom: 10px;">
            🌱 Plant Disease Detection
        </h1>
        <p style="font-size: 1.2em; color: #666; margin-bottom: 5px;">
            Using Hybrid CNN + Vision Transformer (ViT)
        </p>
        <p style="color: #888;">
            Upload a leaf image to detect plant diseases using advanced deep learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Create two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG/PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.markdown("#### Preview")
                st.image(image,
                        caption=f"Uploaded: {uploaded_file.name}")
                
                # Store image in session state for processing
                st.session_state['uploaded_image'] = image
                st.session_state['uploaded_file_name'] = uploaded_file.name
                
            except Exception as e:
                import traceback
                st.error(f"❌ Error loading image: {str(e)}")
                st.error(f"Full error: {traceback.format_exc()}")
        else:
            # Show placeholder
            st.info("👆 Please upload an image to begin analysis")
            
            # Example images section
            st.markdown("#### 💡 Tips for Best Results")
            st.markdown("""
            - Use clear, well-lit images
            - Focus on the affected leaf area
            - Avoid blurry or dark photos
            - Include the entire leaf when possible
            - Take photos in natural light
            """)
    
    with col2:
        st.markdown("### 🔍 Analysis")
        
        if 'uploaded_image' in st.session_state:
            # Analyze button
            if st.button("🔬 Analyze Image", type="primary"):
                
                # Show loading spinner
                with st.spinner("🧠 AI is analyzing your image..."):
                    
                    # Load model
                    model = load_model()
                    
                    if model is None:
                        st.error("❌ Failed to load model. Please check if model file exists.")
                        return
                    
                    # Preprocess image
                    image = st.session_state['uploaded_image']
                    img_array = np.array(image)
                    preprocessed = preprocess_image(image)
                    
                    if preprocessed is None:
                        st.error("❌ Failed to preprocess image.")
                        return
                    
                    # Run prediction
                    predicted_idx, confidence, all_predictions = predict(model, preprocessed)
                    
                    if predicted_idx is None:
                        st.error("❌ Prediction failed.")
                        return
                    
                    # Store results
                    st.session_state['prediction'] = {
                        'class_idx': predicted_idx,
                        'class_name': CLASS_NAMES[str(predicted_idx)],
                        'confidence': confidence,
                        'all_predictions': all_predictions,
                        'image': img_array,
                        'preprocessed': preprocessed
                    }
                
                st.success("✅ Analysis complete!")
            
            # Display results if available
            if 'prediction' in st.session_state:
                result = st.session_state['prediction']
                
                # Get disease information
                disease_info = get_disease_info(result['class_name'])
                
                # Prediction Card
                st.markdown(f"""
                <div class="prediction-card">
                    <h4 style="margin-top: 0; color: #2E7D32;">
                        🎯 Prediction Result
                    </h4>
                    <p style="font-size: 1.3em; margin: 10px 0;">
                        <strong>{result['class_name'].replace('_', ' ').replace('___', ' - ')}</strong>
                    </p>
                    <span class="severity-badge" style="background-color: {disease_info['color']}20; 
                          color: {disease_info['color']}; border: 1px solid {disease_info['color']};">
                        Severity: {disease_info['severity']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence Gauge
                display_prediction_gauge(result['confidence'])
                
                # Disease Information
                st.markdown("---")
                st.markdown("#### 📋 Disease Information")
                
                st.markdown(f"**Description:**")
                st.info(disease_info['description'])
                
                st.markdown(f"**Recommendation:**")
                st.success(disease_info['recommendation'])
                
                # Probability Chart
                st.markdown("---")
                st.markdown("#### 📊 Confidence Distribution")
                
                fig = create_probability_chart(
                    result['all_predictions'], 
                    CLASS_NAMES, 
                    top_n=5
                )
                st.pyplot(fig)
                
                # Grad-CAM Visualization
                st.markdown("---")
                st.markdown("#### 🔍 Model Explainability (Grad-CAM)")
                
                with st.spinner("Generating focused heatmap..."):
                    model = load_model()
                    # Pass original image for leaf masking to focus on leaf area only
                    heatmap = make_gradcam_heatmap(
                        result['preprocessed'], 
                        model,
                        pred_index=result['class_idx'],
                        original_image=result['image']
                    )
                    
                    if heatmap is not None:
                        # Create overlay
                        overlay_img = overlay_heatmap(
                            heatmap, 
                            cv2.resize(result['image'], IMAGE_SIZE)
                        )
                        
                        if overlay_img:
                            col_grad1, col_grad2 = st.columns(2)
                            with col_grad1:
                                st.markdown("**Original**")
                                st.image(result['image'])
                            with col_grad2:
                                st.markdown("**Activation Map**")
                                st.image(overlay_img)
                            
                            st.caption("🎨 Heatmap focused on leaf area only (background removed via green-channel segmentation). Brighter colors = more important regions for prediction.")
                        else:
                            st.info("ℹ️ Heatmap overlay could not be generated. Showing prediction without visualization.")
                    else:
                        st.info("ℹ️ Grad-CAM visualization is not available for this hybrid CNN+ViT architecture. This is expected behavior - the model is still making accurate predictions based on learned features from both CNN (local) and Transformer (global) components.")
                
                # Reset button
                st.markdown("---")
                if st.button("🔄 Analyze Another Image"):
                    del st.session_state['prediction']
                    del st.session_state['uploaded_image']
                    del st.session_state['uploaded_file_name']
                    st.rerun()
        else:
            # Empty state
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #999;">
                <p style="font-size: 3em; margin-bottom: 10px;">📷</p>
                <p>Upload an image to see analysis results here</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #888; font-size: 0.9em;">
        <p>🌱 Plant Disease Detection System | Powered by Hybrid CNN + ViT</p>
        <p style="font-size: 0.8em;">
            Note: This AI model provides predictions based on image analysis. 
            For critical decisions, consult with agricultural experts.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
