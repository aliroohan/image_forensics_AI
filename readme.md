# Image Forensic & Fraud Detection Tool

A comprehensive tool for analyzing images and detecting potential manipulations or fraudulent alterations using various forensic techniques. This MVP (Minimum Viable Product) demonstrates multiple image analysis methods to provide a detailed report on the likelihood of image manipulation.

## 🔍 Features

- **Error Level Analysis (ELA)**: Detects inconsistencies in JPEG compression levels that may indicate manipulation
- **Clone Detection**: Identifies duplicated regions within an image that suggest copy-paste operations
- **Noise Analysis**: Examines noise patterns to detect inconsistencies between different areas of the image
- **EXIF Metadata Analysis**: Extracts and analyzes image metadata for signs of editing or manipulation
- **AI-based Manipulation Detection**: Provides a heatmap highlighting regions with a high probability of manipulation
- **Combined Analysis Report**: Aggregates results from all techniques to provide an overall manipulation likelihood score

## 🏗️ Architecture

The application consists of two main components:

1. **FastAPI Backend** (`backend.py`): Provides image analysis endpoints and core forensic functionality
2. **Gradio Frontend** (`main.py`): Delivers a user-friendly web interface to interact with the API

```
┌─────────────┐      HTTP       ┌─────────────┐
│   Gradio    │◄─── Requests ───►│   FastAPI   │
│  Frontend   │                 │   Backend   │
└─────────────┘                 └─────────────┘
      ▲                                ▲
      │                                │
      └────────── User ────────────────┘
```

## 🚀 Installation



1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## 🔧 Usage

1. Start the FastAPI backend:
```bash
uvicorn backend:app --reload
```

2. In a separate terminal, start the Gradio frontend:
```bash
python main.py
```


## 🧠 How It Works

### Error Level Analysis (ELA)
ELA works by saving the image at a specific quality level (e.g., 90%) and then comparing this re-saved image with the original. Areas with high error levels often indicate manipulation, as different parts of a manipulated image may have different compression histories.

### Clone Detection
This technique identifies areas within the image that have been duplicated (copy-pasted). It works by dividing the image into small blocks and comparing the similarity between all blocks. Highly similar blocks that are not adjacent to each other often indicate cloning operations.

### Noise Analysis
Every image contains a certain level of noise. When images are manipulated, the noise pattern can become inconsistent. This technique amplifies the noise in an image to reveal these inconsistencies.

### EXIF Metadata Analysis
Images contain metadata that includes information about the camera, software used, and editing history. This analysis examines this metadata for signs of editing such as Photoshop usage or timestamp inconsistencies.

### AI Detection Heatmap
This technique currently simulates a pre-trained AI model that evaluates the likelihood of manipulation for different regions of the image, highlighting suspicious areas in red. In a production environment, this would be replaced with an actual trained deep learning model.

## 🧪 API Endpoints

The FastAPI backend provides several endpoints:

- `/api/analyze_image`: Main endpoint that performs complete analysis
- `/api/detect_clones`: Detects cloned/copy-pasted regions
- `/api/error_level_analysis`: Performs Error Level Analysis
- `/api/extract_exif_metadata`: Extracts and analyzes image metadata
- `/api/noise_analysis`: Analyzes noise patterns
- `/api/manipulation_likelihood`: Evaluates overall manipulation probability

## 📊 Gradio Interface

The Gradio frontend provides:
- Image upload capability
- Multiple tabs for viewing different analysis results
- EXIF metadata display
- Manipulation probability indicator
- Detailed analysis report


