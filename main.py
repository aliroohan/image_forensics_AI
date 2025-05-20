import gradio as gr
import requests
import json
import io
import base64
from PIL import Image

# FastAPI backend URL
API_URL = "http://localhost:8000"

#############################
# HELPER FUNCTIONS
#############################

def base64_to_pil(base64_str):
    """Convert base64 string to PIL image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def upload_image(image, endpoint):
    """Upload an image to the specified API endpoint"""
    # Save image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Send to API
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post(f"{API_URL}{endpoint}", files=files)
    
    # Return the JSON response
    return response.json()

#############################
# API INTERFACE FUNCTIONS
#############################

def analyze_image(image):
    """Main function that sends the image to the API for analysis"""
    if image is None:
        return {
            original_image: None,
            ela_image: None,
            noise_image: None,
            heatmap_image: None,
            clone_image: None,
            exif_data: "{}",
            analysis_results: "Please upload an image first.",
            probability_slider: 0
        }
        
    # Send to API for full analysis
    try:
        response = upload_image(image, "/api/analyze_image")
        
        # Process results
        return {
            original_image: image,
            ela_image: base64_to_pil(response["ela_image"]),
            noise_image: base64_to_pil(response["noise_image"]),
            heatmap_image: base64_to_pil(response["heatmap_image"]),
            clone_image: base64_to_pil(response["clone_image"]),
            exif_data: json.dumps(response["exif_data"], indent=2),
            analysis_results: response["analysis_text"],
            probability_slider: response["manipulation_probability"]
        }
    except Exception as e:
        return {
            original_image: image,
            ela_image: None,
            noise_image: None,
            heatmap_image: None,
            clone_image: None,
            exif_data: f"Error: {str(e)}",
            analysis_results: f"Error occurred during analysis: {str(e)}",
            probability_slider: 0
        }


#############################
# GRADIO INTERFACE
#############################

with gr.Blocks(title="Image Forensic & Fraud Detection Tool - MVP Demo") as demo:
    gr.Markdown("""
    # Image Forensic & Fraud Detection Tool

    
    Upload an image to analyze it for potential manipulation using various forensic techniques.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image for Analysis")
            analyze_button = gr.Button("Analyze Image", variant="primary")
            
            gr.Markdown("### Manipulation Probability")
            probability_slider = gr.Slider(
                minimum=0, maximum=1, value=0, 
                label="Manipulation Probability", 
                interactive=False
            )
            
            gr.Markdown("### EXIF Metadata")
            exif_data = gr.Code(language="json", label="EXIF Data", lines=10)
            
        with gr.Column(scale=2):
            with gr.Tab("Analysis Results"):
                analysis_results = gr.Markdown()
                
            with gr.Tab("Original Image"):
                original_image = gr.Image(type="pil", label="Original Image")
                
            with gr.Tab("Error Level Analysis (ELA)"):
                gr.Markdown("""
                Error Level Analysis reveals differences in compression levels. Areas with different compression levels 
                often indicate modifications. Brighter regions in the visualization suggest potential manipulations.
                """)
                ela_image = gr.Image(type="pil", label="ELA Result")
                
            with gr.Tab("Noise Analysis"):
                gr.Markdown("""
                Noise Analysis examines the noise patterns in the image. Inconsistent noise patterns often indicate 
                areas that have been manipulated or added from different sources.
                """)
                noise_image = gr.Image(type="pil", label="Noise Pattern Analysis")
                
            with gr.Tab("Clone Detection"):
                gr.Markdown("""
                Clone Detection identifies duplicated areas within the image. Red and blue rectangles highlight 
                matching regions that may indicate copy-paste manipulation.
                """)
                clone_image = gr.Image(type="pil", label="Clone Detection Result")
                
            with gr.Tab("AI Detection Heatmap"):
                gr.Markdown("""
                This heatmap highlights regions identified by our AI model as potentially manipulated.
                Red areas indicate suspicious regions with a higher likelihood of manipulation.
                """)
                heatmap_image = gr.Image(type="pil", label="AI-Detected Suspicious Regions")
    
    # Set up event handlers
    analyze_button.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[
            original_image, 
            ela_image, 
            noise_image, 
            heatmap_image, 
            clone_image,
            exif_data, 
            analysis_results,
            probability_slider
        ]
    )
    
# Launch the app
if __name__ == "__main__":
    demo.launch()