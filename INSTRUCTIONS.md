# Plant Disease Detection - Streamlit Web Application

A complete, production-ready web application for detecting plant diseases using a Hybrid CNN + Vision Transformer (ViT) deep learning model.

---

## Features

- **38 Disease Categories**: Detects diseases across 14 different plant types
- **Hybrid AI Model**: Combines CNN local feature extraction with ViT global pattern recognition
- **Real-time Analysis**: Instant predictions with confidence scores
- **Grad-CAM Visualization**: See what the model focuses on for explainability
- **Beautiful UI**: Modern, responsive design with Streamlit
- **Comprehensive Information**: Disease descriptions and treatment recommendations

---

## Supported Plants & Diseases

| Plant | Diseases |
|-------|----------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🍊 Orange | Citrus Greening (HLB) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍓 Raspberry | Healthy |
| 🫘 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Navigate to the App Directory

```bash
cd c:\Users\Nishanth\Desktop\plant-disease-prediction-cnn-deep-leanring-project-main\plant-disease-prediction-cnn-deep-leanring-project-main\app
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv_streamlit
venv_streamlit\Scripts\activate

# macOS/Linux
python3 -m venv venv_streamlit
source venv_streamlit/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

Or install manually:

```bash
pip install streamlit==1.40.0 tensorflow==2.15.0 opencv-python==4.8.1.78 matplotlib==3.7.2 numpy==1.24.3 Pillow==10.0.0
```

### Step 4: Verify Model File

Ensure the model file exists at:
```
app/trained_model/hybrid_vit_cnn_plant_disease_model.keras
```

---

## Running the Application

### Method 1: Using Streamlit CLI

```bash
streamlit run app.py
```

### Method 2: Using Python Module

```bash
python -m streamlit run app.py
```

### Method 3: Using Run Script (Windows)

Create a `run_app.bat` file:
```batch
@echo off
cd /d "%~dp0"
call venv_streamlit\Scripts\activate
streamlit run app.py
pause
```

Then double-click to run.

---

## Usage Guide

### 1. Launch the Application
After running the command, your browser should automatically open at `http://localhost:8501`

### 2. Upload an Image
- Click the "Browse files" button in the left panel
- Select a clear image of a plant leaf (JPG/PNG)
- Supported formats: `.jpg`, `.jpeg`, `.png`

### 3. Analyze
- Click the **"🔬 Analyze Image"** button
- Wait for the AI to process (usually 1-3 seconds)

### 4. View Results
The right panel will show:
- **Predicted Disease**: Top classification result
- **Confidence Score**: How certain the model is (percentage)
- **Disease Description**: Information about the detected disease
- **Recommendations**: Treatment and management advice
- **Top 5 Predictions**: Bar chart of alternative possibilities
- **Grad-CAM Heatmap**: Visual explanation of model focus areas

### 5. Start New Analysis
Click **"🔄 Analyze Another Image"** to clear and upload a new image.

---

## Project Structure

```
app/
├── app.py                          # Main Streamlit application
├── requirements_streamlit.txt      # Python dependencies
├── class_indices.json               # Disease class mappings
├── trained_model/
│   └── hybrid_vit_cnn_plant_disease_model.keras  # Pre-trained model
├── uploads/                         # Uploaded images (optional)
└── venv_streamlit/                  # Virtual environment
```

---

## Model Architecture

### Hybrid CNN + Vision Transformer (ViT)

The model combines the strengths of both architectures:

1. **CNN (Convolutional Neural Network)**
   - Extracts local spatial features from leaf textures
   - Captures edges, spots, and color patterns
   - Efficient hierarchical feature learning

2. **Vision Transformer (ViT)**
   - Models global dependencies across the entire image
   - Captures long-range relationships between symptoms
   - Self-attention mechanism for important regions

3. **Fusion Layer**
   - Combines CNN local features with ViT global features
   - Produces robust representations for accurate classification

**Input**: 128×128 RGB images  
**Output**: 38 disease classes  
**Accuracy**: >95% on test set

---

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Verify `hybrid_vit_cnn_plant_disease_model.keras` exists in `trained_model/` folder

### Issue: "ModuleNotFoundError"
**Solution**: Install missing package: `pip install [package_name]`

### Issue: "CUDA out of memory"
**Solution**: The model will automatically use CPU if GPU memory is insufficient

### Issue: "Port 8501 already in use"
**Solution**: Run on different port: `streamlit run app.py --server.port 8502`

### Issue: Grad-CAM not working
**Solution**: Some model architectures may not support Grad-CAM. The app will gracefully handle this.

---

## Performance Optimization

### For Faster Predictions:
1. Use GPU if available (install `tensorflow-gpu`)
2. Reduce image size before upload
3. Use batch processing for multiple images

### For Production Deployment:
```bash
# Run with optimizations
streamlit run app.py --server.maxUploadSize 10 --server.enableCORS false
```

---

## Deployment Options

### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Cloud Deployment
- **Streamlit Cloud**: Connect GitHub repo to share.streamlit.io
- **Heroku**: Use `Procfile` with web process
- **AWS/Azure/GCP**: Deploy as containerized application

---

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (responsive design)

---

## API Usage (Optional)

The app can be extended to provide API endpoints:

```python
# Example: Programmatic prediction
from app import load_model, preprocess_image, predict
import numpy as np
from PIL import Image

model = load_model()
image = Image.open('leaf.jpg')
processed = preprocess_image(image)
predicted_idx, confidence, _ = predict(model, processed)
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

---

## License

This project is for educational and research purposes.

---

## Contact & Support

For issues or questions:
- Create an issue in the repository
- Contact the development team

---

## Acknowledgments

- Dataset: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Framework: TensorFlow/Keras, Streamlit
- Icons: Emoji set

---

**Happy Plant Disease Detection! 🌱🔬**
