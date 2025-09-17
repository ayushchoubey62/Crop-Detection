import os
import io
import json
import logging
import numpy as np
import cv2
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from flask import Flask, render_template, request, jsonify, send_file, url_for, make_response
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from fpdf import FPDF # For PDF generation
import tempfile # Import tempfile for creating temporary files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
MODELS_FOLDER = 'models'
FEEDBACK_FOLDER = 'feedback'
HISTORY_FILE = 'app_data.json' # Using a single JSON for history and settings
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit for uploads

# --- Global Variables for Models and Data (Loaded once on app startup) ---
disease_model = None
crop_classifier = None # Placeholder for a potential crop classifier model
class_labels = None
app_data = { # Initialize with default values
    'dark_mode': False,
    'diagnosis_history': [],
    'recent_images': []
}

# --- Standardized Disease Information Dictionary ---
# Keys here MUST exactly match the 'predicted_label' from the model output,
# except for 'healthy' which is a generic key.
disease_info = {
    "Apple_Apple_scab": {
        "description": "Apple scab is a common fungal disease that affects apple and crabapple trees. It causes olive-green to brown spots on leaves, fruit, and twigs. Severe infections can lead to premature leaf drop and deformed fruit.",
        "symptoms": "Olive-green to brown spots on leaves, often with a velvety texture. Spots on fruit are dark, circular, and may become corky or cracked. Twig lesions can also occur."
    },
    "Apple_Black_rot": {
        "description": "Black rot is a fungal disease that affects apple trees, causing lesions on leaves, cankers on branches, and a characteristic black rot on fruit. It can lead to significant yield losses.",
        "symptoms": "Leaf spots are purplish with a brown center. Cankers on branches are sunken and discolored. Fruit rot begins as a brown spot that rapidly expands and turns black, often with concentric rings."
    },
    "Apple_Cedar_apple_rust": {
        "description": "Cedar-apple rust is a fungal disease that requires two hosts: apple/crabapple and cedar/juniper. It causes bright orange spots on apple leaves and can deform fruit.",
        "symptoms": "Bright orange-yellow spots on apple leaves, often with small black dots (spermagonia) in the center. Galls may form on cedar trees, producing gelatinous orange horns in wet weather."
    },
    "Blueberry_healthy": { # This will map to 'healthy' key
        "description": "The blueberry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy berry development."
    },
    "Cherry_Powdery_mildew": {
        "description": "Powdery mildew is a fungal disease that appears as white, powdery patches on the surface of leaves, stems, and sometimes fruit. Severe infections can stunt growth and reduce yield.",
        "symptoms": "White, powdery spots on leaves, stems, and fruit. Leaves may curl, distort, or turn yellow. Young leaves are often more susceptible."
    },
    "Cherry_healthy": { # This will map to 'healthy' key
        "description": "The cherry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development."
    },
    "Corn_Cercospora_leaf_spot_Gray_leaf_spot": {
        "description": "Cercospora leaf spot (also known as Gray leaf spot) is a fungal disease of corn characterized by rectangular, gray-to-tan lesions on leaves. Severe infections can lead to significant yield loss.",
        "symptoms": "Long, narrow, rectangular lesions (1-2 inches long) that are gray to tan in color. Lesions are typically restricted by leaf veins. May appear water-soaked initially."
    },
    "Corn_Common_rust": {
        "description": "Common rust is a fungal disease of corn characterized by the formation of reddish-brown pustules on leaves. Severe infections can reduce photosynthetic area and impact yield.",
        "symptoms": "Small, cinnamon-brown to reddish-brown pustules on both upper and lower leaf surfaces. Pustules may rupture, releasing powdery spores."
    },
    "Corn_Northern_Leaf_Blight": {
        "description": "Northern Leaf Blight is a fungal disease of corn that causes long, elliptical, gray-green lesions on leaves. It can significantly reduce photosynthetic area and affect grain fill.",
        "symptoms": "Long, elliptical, gray-green to tan lesions on leaves, typically 1 to 6 inches long. Lesions may coalesce, blighting large areas of the leaf."
    },
    "Corn_healthy": { # This will map to 'healthy' key
        "description": "The corn plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy stalk and ear development."
    },
    "Grape_Black_rot": {
        "description": "Black rot is a destructive fungal disease of grapes that affects leaves, shoots, and fruit. It causes characteristic black, shriveled mummies on the fruit.",
        "symptoms": "Small, circular, reddish-brown spots on leaves that enlarge and turn tan with dark borders. Black, shriveled, raisin-like fruit mummies. Lesions on shoots and tendrils."
    },
    "Grape_Esca_Black_Measles": {
        "description": "Esca (also known as Black Measles) is a complex of fungal diseases affecting grapevines, leading to wood decay and foliar symptoms. It can cause sudden vine collapse or chronic decline.",
        "symptoms": "Foliar symptoms include interveinal chlorosis (yellowing) followed by necrosis (browning), often with a 'tiger-stripe' pattern. Fruit may develop dark spots and shrivel. Wood symptoms include dark streaking and decay."
    },
    "Grape_Leaf_blight": {
        "description": "Grape leaf blight is a general term for various conditions causing browning and death of leaf tissue. It can be caused by fungi, bacteria, or environmental factors.",
        "symptoms": "Irregular brown spots or patches on leaves, often starting at the margins or between veins. Affected areas may dry out and become brittle. Severe cases can lead to defoliation."
    },
    "Grape_healthy": { # This will map to 'healthy' key
        "description": "The grape plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy vine and fruit development."
    },
    "Orange_Haunglongbing_Citrus_greening": {
        "description": "Huanglongbing (HLB), also known as Citrus Greening, is a devastating bacterial disease of citrus trees spread by psyllids. It causes yellowing of leaves, misshapen fruit, and eventual tree decline.",
        "symptoms": "Asymmetrical blotchy mottling or yellowing of leaves, often resembling nutrient deficiencies but not symmetrical. Small, lopsided, green-bottomed fruit that taste bitter. Premature fruit drop."
    },
    "Peach_Bacterial_spot": {
        "description": "Bacterial spot is a common disease of peaches caused by Xanthomonas arboricola pv. pruni. It causes small, angular spots on leaves, lesions on twigs, and fruit spots.",
        "symptoms": "Small, angular, water-soaked spots on leaves that turn purplish-brown and may drop out, giving a 'shot-hole' appearance. Sunken, dark lesions on twigs and fruit. Fruit spots are dark, pitted, and may crack."
    },
    "Peach_healthy": { # This will map to 'healthy' key
        "description": "The peach plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development."
    },
    "Pepper_bell_Bacterial_spot": {
        "description": "Bacterial spot is a common and destructive disease of pepper plants caused by Xanthomonas bacteria. It leads to dark, water-soaked spots on leaves and fruit.",
        "symptoms": "Small, dark, water-soaked spots on leaves that may develop yellow halos. Spots on fruit are dark, raised, and scabby. Severe infections can cause leaf yellowing and defoliation."
    },
    "Pepper_bell_healthy": { # This will map to 'healthy' key
        "description": "The bell pepper plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development."
    },
    "Potato_Early_blight": {
        "description": "Early blight is a fungal disease affecting potato and tomato plants. It causes dark, concentric ringed spots on older leaves, leading to defoliation and reduced yield.",
        "symptoms": "Dark brown to black spots with concentric rings (like a target) on older leaves. Lesions may also appear on stems and tubers. Yellowing of tissue around spots."
    },
    "Potato_Late_blight": {
        "description": "Late blight is a devastating disease of potato and tomato caused by a water mold (Phytophthora infestans). It can rapidly destroy entire crops under cool, wet conditions.",
        "symptoms": "Irregular, water-soaked, dark green to brown lesions on leaves, often starting at the leaf tips or edges. White, fuzzy fungal growth may be visible on the undersides of leaves during humid conditions. Brown, firm rot on tubers."
    },
    "Potato_healthy": { # This will map to 'healthy' key
        "description": "The potato plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy tuber development."
    },
    "Raspberry_healthy": { # This will map to 'healthy' key
        "description": "The raspberry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy berry development."
    },
    "Soybean_healthy": { # This will map to 'healthy' key
        "description": "The soybean plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy pod development."
    },
    "Squash_Powdery_mildew": {
        "description": "Powdery mildew is a common fungal disease affecting squash and other cucurbits. It appears as white, powdery spots on leaves and stems, reducing photosynthesis and yield.",
        "symptoms": "White, powdery patches on the upper and lower surfaces of leaves and stems. Leaves may turn yellow, then brown, and eventually die. Reduced fruit size and quality."
    },
    "Strawberry_Leaf_scorch": {
        "description": "Leaf scorch is a fungal disease of strawberries causing purplish spots on leaves that eventually enlarge and coalesce, leading to a 'scorched' appearance.",
        "symptoms": "Small, purplish spots on leaves that enlarge into irregular, reddish-brown blotches. Margins of the spots may turn brown or reddish. Severe infections can cause leaves to dry up and curl."
    },
    "Strawberry_healthy": { # This will map to 'healthy' key
        "description": "The strawberry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot is a common and destructive disease of tomato and pepper caused by several species of Xanthomonas bacteria. It leads to dark, water-soaked spots on leaves and fruit.",
        "symptoms": "Small, dark, water-soaked spots on leaves that may develop yellow halos. Spots on fruit are dark, raised, and scabby. Severe infections can cause leaf yellowing and defoliation."
    },
    "Tomato_Early_blight": {
        "description": "Early blight is a fungal disease affecting tomato and potato plants. It causes dark, concentric ringed spots on older leaves, leading to defoliation and reduced yield.",
        "symptoms": "Dark brown to black spots with concentric rings (like a target) on older leaves. Lesions may also appear on stems and fruit. Yellowing of tissue around spots."
    },
    "Tomato_Late_blight": {
        "description": "Late blight is a devastating disease of tomato and potato caused by a water mold (Phytophthora infestans). It can rapidly destroy entire crops under cool, wet conditions.",
        "symptoms": "Irregular, water-soaked, dark green to brown lesions on leaves, often starting at the leaf tips or edges. White, fuzzy fungal growth may be visible on the undersides of leaves during humid conditions. Brown, firm rot on fruit."
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold is a fungal disease of tomatoes, especially common in humid conditions. It causes velvety, olive-green to brown patches on the undersides of leaves.",
        "symptoms": "Yellowish spots on the upper leaf surface, with olive-green to brown, velvety fungal growth on the corresponding undersides. Leaves may curl, dry up, and fall off."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot is a common fungal disease of tomatoes. It causes numerous small, circular spots on older leaves, often with dark borders and tiny black dots in the center.",
        "symptoms": "Numerous small, circular spots (1/8 to 1/4 inch) on older leaves. Spots have dark brown borders and tan to gray centers, often with tiny black specks (pycnidia) in the middle. Severe infections lead to defoliation."
    },
    "Tomato_Spider_mites _Two_spotted_spider_mite": { # This key is correct as is, matching the class_labels
        "description": "Spider mites are tiny pests that feed on plant cells, causing stippling and yellowing of leaves. Heavy infestations can lead to webbing and severe plant damage.",
        "symptoms": "Tiny yellow or white stippling (pinprick dots) on leaves. Yellowing, bronzing, or drying of leaves. Fine webbing on the undersides of leaves or between stems. Mites are barely visible to the naked eye."
    },
    "Tomato_Target_Spot": {
        "description": "Target spot is a fungal disease of tomatoes causing circular lesions with concentric rings, resembling a target. It affects leaves, stems, and fruit.",
        "symptoms": "Small, circular, water-soaked spots on leaves that enlarge and develop concentric rings, turning brown to black. A yellow halo may surround the spots. Lesions can also appear on stems and fruit."
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": { # Key now matches full predicted_label
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease of tomatoes transmitted by whiteflies. It causes severe stunting and yellowing of leaves.",
        "symptoms": "Severe stunting of plants. Upward curling and yellowing of leaf margins, especially on younger leaves. Leaves become thickened and brittle. Flowers may drop, and fruit set is severely reduced."
    },
    "Tomato_Tomato_mosaic_virus": { # Key now matches full predicted_label
        "description": "Tomato Mosaic Virus (ToMV) is a highly contagious viral disease that causes mosaic patterns, mottling, and distortion of leaves in tomato plants.",
        "symptoms": "Light and dark green mosaic patterns or mottling on leaves. Leaves may be distorted, puckered, or fern-like. Stunting of plants and reduced fruit size or quality."
    },
    "healthy": { # Single generic healthy entry for all healthy crops
        "description": "The plant appears healthy with no visible signs of disease. Maintaining good cultural practices is key to preventing future issues.",
        "symptoms": "Uniform green coloration, no spots, lesions, or deformities. Leaves are turgid and vibrant."
    }
}

# --- Standardized Treatment Suggestions Dictionary ---
# Keys here MUST exactly match the 'predicted_label' from the model output,
# except for 'healthy' which is a generic key.
treatment_suggestions_data = {
    "Apple_Apple_scab": [
        "Apply fungicides containing myclobutanil or sulfur",
        "Remove and destroy infected leaves and fruit",
        "Improve air circulation through pruning"
    ],
    "Apple_Black_rot": [
        "Prune out infected plant parts (canes, fruit)",
        "Apply fungicides like captan or mancozeb during susceptible stages",
        "Ensure good air circulation and sunlight exposure"
    ],
    "Apple_Cedar_apple_rust": [
        "Remove nearby cedar trees (alternate host)",
        "Apply fungicides such as myclobutanil",
        "Plant resistant apple varieties"
    ],
    "Blueberry_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Cherry_Powdery_mildew": [
        "Apply sulfur or potassium bicarbonate",
        "Reduce humidity around plants",
        "Use resistant varieties when available",
        "Prune affected areas to improve air circulation"
    ],
    "Cherry_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Corn_Cercospora_leaf_spot_Gray_leaf_spot": [
        "Apply fungicides containing chlorothalonil or copper",
        "Rotate crops to non-host plants",
        "Remove and destroy infected plant debris"
    ],
    "Corn_Common_rust": [
        "Use resistant varieties",
        "Apply fungicides if severe, such as those with propiconazole or azoxystrobin",
        "Practice good sanitation to remove rust spores"
    ],
    "Corn_Northern_Leaf_Blight": [
        "Use resistant varieties",
        "Apply fungicides if severe, typically those containing strobilurins or triazoles",
        "Practice crop rotation and residue management"
    ],
    "Corn_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Grape_Black_rot": [
        "Apply fungicides containing mancozeb, myclobutanil, or captan",
        "Sanitation: Remove and destroy all mummified berries and infected plant parts",
        "Pruning: Prune to improve air circulation and canopy drying"
    ],
    "Grape_Esca_Black_Measles": [
        "Prune out diseased wood during dry periods",
        "Apply protective fungicidal paints to pruning wounds",
        "Improve vineyard hygiene to reduce inoculum"
    ],
    "Grape_Leaf_blight": [
        "Apply fungicides (e.g., copper-based or mancozeb)",
        "Improve air circulation around plants by proper spacing and pruning",
        "Remove and dispose of infected leaves"
    ],
    "Grape_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Orange_Haunglongbing_Citrus_greening": [
        "No known cure; remove infected trees to prevent spread",
        "Control citrus psyllid vectors with insecticides",
        "Plant certified disease-free nursery stock"
    ],
    "Peach_Bacterial_spot": [
        "Apply copper-based bactericides",
        "Use disease-free seeds and transplants",
        "Avoid overhead watering and splashing soil onto plants",
        "Practice crop rotation"
    ],
    "Peach_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Pepper_bell_Bacterial_spot": [
        "Apply copper-based bactericides",
        "Use disease-free seeds and transplants",
        "Avoid overhead watering and splashing soil onto plants",
        "Practice crop rotation"
    ],
    "Pepper_bell_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Potato_Early_blight": [
        "Apply copper-based fungicides",
        "Practice crop rotation",
        "Remove and destroy infected plant material"
    ],
    "Potato_Late_blight": [
        "Apply fungicides containing chlorothalonil or copper",
        "Destroy infected plants immediately",
        "Avoid overhead watering"
    ],
    "Potato_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Raspberry_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Soybean_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Squash_Powdery_mildew": [
        "Apply sulfur or potassium bicarbonate",
        "Reduce humidity around plants",
        "Use resistant varieties when available",
        "Prune affected areas to improve air circulation"
    ],
    "Strawberry_Leaf_scorch": [
        "Apply fungicides with captan or myclobutanil",
        "Sanitation: Remove and destroy infected leaves",
        "Mulching: Use mulch to prevent splashing spores"
    ],
    "Strawberry_healthy": [
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Ensure good air circulation and balanced nutrition"
    ],
    "Tomato_Bacterial_spot": [
        "Apply copper-based bactericides",
        "Use disease-free seeds and transplants",
        "Avoid overhead watering and splashing soil onto plants",
        "Practice crop rotation"
    ],
    "Tomato_Early_blight": [
        "Apply copper-based fungicides",
        "Practice crop rotation",
        "Remove and destroy infected plant material"
    ],
    "Tomato_Late_blight": [
        "Apply fungicides containing chlorothalonil or copper",
        "Destroy infected plants immediately",
        "Avoid overhead watering"
    ],
    "Tomato_Leaf_Mold": [
        "Improve air circulation by pruning and spacing plants",
        "Reduce humidity in greenhouses",
        "Apply fungicides if necessary, suchg as chlorothalonil or mancozeb"
    ],
    "Tomato_Septoria_leaf_spot": [
        "Apply fungicides with chlorothalonil or mancozeb",
        "Remove infected leaves and plant debris",
        "Avoid overhead watering to minimize leaf wetness"
    ],
    "Tomato_Spider_mites _Two_spotted_spider_mite": [ # This key is correct as is
        "Apply insecticidal soaps or horticultural oils",
        "Increase humidity around plants (mist foliage)",
        "Introduce predatory mites"
    ],
    "Tomato_Target_Spot": [
        "Apply fungicides containing chlorothalonil or mancozeb",
        "Practice crop rotation and sanitation",
        "Ensure good air circulation"
    ],
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": [ # Key now matches full predicted_label
        "No direct chemical cure for the virus; manage whitefly vectors with insecticides",
        "Use resistant varieties if available",
        "Remove infected plants immediately"
    ],
    "Tomato_Tomato_mosaic_virus": [ # Key now matches full predicted_label
        "No chemical cure; remove and destroy infected plants",
        "Disinfect tools and hands after handling infected plants",
        "Use resistant varieties or certified disease-free seeds"
    ],
    "healthy": [ # Single generic healthy entry for all healthy crops
        "Continue regular monitoring",
        "Maintain good cultural practices (proper watering, fertilization, pruning)",
        "Preventative measures like ensuring good air circulation and balanced nutrition"
    ]
}


# --- Helper Functions ---
def load_app_data():
    global app_data
    default_app_data = {
        'dark_mode': False,
        'diagnosis_history': [],
        'recent_images': []
    }
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                # Ensure loaded_data is a dictionary
                if isinstance(loaded_data, dict):
                    app_data = loaded_data
                    # Ensure dark_mode key exists and is boolean
                    if 'dark_mode' not in app_data or not isinstance(app_data['dark_mode'], bool):
                        app_data['dark_mode'] = False

                    # Filter out non-existent image paths from history to avoid errors
                    app_data['diagnosis_history'] = [
                        item for item in app_data.get('diagnosis_history', [])
                        if 'image_path' not in item or (item['image_path'] and os.path.exists(item['image_path']))
                    ]
                    # Filter recent images, ensuring paths exist
                    app_data['recent_images'] = [
                        path for path in app_data.get('recent_images', [])
                        if os.path.exists(path)
                    ]
                    logging.info("Application data loaded successfully.")
                else:
                    logging.warning(f"Loaded data from {HISTORY_FILE} is not a dictionary. Starting with default settings.")
                    app_data = default_app_data
        else:
            app_data = default_app_data # Initialize if file doesn't exist
    except (json.JSONDecodeError, Exception) as e:
        logging.error(f"Error loading application data from {HISTORY_FILE}: {e}. Starting with default settings.", exc_info=True)
        app_data = default_app_data

def save_app_data():
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(app_data, f, indent=4)
        logging.info(f"Application data saved successfully to {HISTORY_FILE}.")
    except Exception as e:
        logging.error(f"Error saving application data to {HISTORY_FILE}: {e}", exc_info=True)

def load_models():
    global disease_model, class_labels
    model_path = os.path.join(MODELS_FOLDER, "my_model.keras")
    try:
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}. Please ensure 'my_model.keras' is in the '{MODELS_FOLDER}' directory.")
            return False

        disease_model = load_model(model_path)
        # Warm-up prediction
        dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
        _ = disease_model.predict(dummy_input, verbose=0)
        logging.info("Disease detection model loaded and warmed up successfully.")

        # Define class labels - make sure these match your model's output
        class_labels = {
            'Apple_Apple_scab': 0, 'Apple_Black_rot': 1, 'Apple_Cedar_apple_rust': 2,
            'Apple_healthy': 3, 'Blueberry_healthy': 4, 'Cherry_Powdery_mildew': 5,
            'Cherry_healthy': 6, 'Corn_Cercospora_leaf_spot_Gray_leaf_spot': 7,
            'Corn_Common_rust': 8, 'Corn_Northern_Leaf_Blight': 9, 'Corn_healthy': 10,
            'Grape_Black_rot': 11, 'Grape_Esca_Black_Measles': 12, 'Grape_Leaf_blight': 13,
            'Grape_healthy': 14, 'Orange_Haunglongbing_Citrus_greening': 15,
            'Peach_Bacterial_spot': 16, 'Peach_healthy': 17, 'Pepper_bell_Bacterial_spot': 18,
            'Pepper_bell_healthy': 19, 'Potato_Early_blight': 20, 'Potato_Late_blight': 21,
            'Potato_healthy': 22, 'Raspberry_healthy': 23, 'Soybean_healthy': 24,
            'Squash_Powdery_mildew': 25, 'Strawberry_Leaf_scorch': 26, 'Strawberry_healthy': 27,
            'Tomato_Bacterial_spot': 28, 'Tomato_Early_blight': 29, 'Tomato_Late_blight': 30,
            'Tomato_Leaf_Mold': 31, 'Tomato_Septoria_leaf_spot': 32,
            'Tomato_Spider_mites _Two_spotted_spider_mite': 33, 'Tomato_Target_Spot': 34,
            'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato_Tomato_mosaic_virus': 36,
            'Tomato_healthy': 37
        }
        logging.info("Class labels defined.")
        return True
    except Exception as e:
        logging.critical(f"Failed to load machine learning model: {e}", exc_info=True)
        return False

def is_non_crop_image_heuristic(img_bytes):
    """
    Enhanced heuristic approach for non-crop detection based on image properties.
    This is a fallback if no ML crop classifier is available.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        if img.size[0] < 100 or img.size[1] < 100:
            logging.info(f"Heuristic non-crop detection: True (Reason: Image too small - {img.size[0]}x{img.size[1]})")
            return True

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        green_pixels_mask = (hsv[:,:,0] > 30) & (hsv[:,:,0] < 90) & (hsv[:,:,1] > 40)
        green_ratio = np.sum(green_pixels_mask) / (img.size[0] * img.size[1])

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        avg_brightness = np.mean(gray)
        contrast = gray.std()

        # Relaxed thresholds for heuristic detection
        if (green_ratio < 0.05 or # Lowered from 0.10
            laplacian_var < 5 or  # Lowered from 10
            avg_brightness < 10 or
            avg_brightness > 245 or
            contrast < 10):       # Lowered from 15
            logging.info(f"Heuristic non-crop detection: True (Green Ratio: {green_ratio:.2f}, Laplacian Var: {laplacian_var:.2f}, Brightness: {avg_brightness:.2f}, Contrast: {contrast:.2f})")
            return True

        logging.info(f"Heuristic non-crop detection: False (Green Ratio: {green_ratio:.2f}, Laplacian Var: {laplacian_var:.2f}, Brightness: {avg_brightness:.2f}, Contrast: {contrast:.2f})")
        return False

    except Exception as e:
        logging.error(f"Heuristic non-crop detection failed: {e}", exc_info=True)
        return False # Default to assuming it's a crop if heuristic detection fails.

def diagnose_disease(img_bytes):
    if disease_model is None or class_labels is None:
        logging.error("diagnose_disease: Disease detection model or class labels not loaded.")
        return {"error": "Disease detection model not loaded."}

    # Perform non-crop detection first
    if is_non_crop_image_heuristic(img_bytes):
        logging.info("diagnose_disease: Heuristic detected non-crop image.")
        return {"error": "Uploaded image doesn't appear to be a crop leaf image. Please upload a clearer image of a single crop leaf."}

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((150, 150), Image.Resampling.LANCZOS) # Resize for model input
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = disease_model.predict(img_array, verbose=0)

        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        predicted_label = list(class_labels.keys())[predicted_class_idx]
        logging.info(f"diagnose_disease: Predicted label from model: '{predicted_label}'")
        
        # Determine the key for dictionary lookup and human-readable display name
        if 'healthy' in predicted_label.lower():
            disease_key_for_lookup = "healthy"
            display_disease_name = "Healthy"
        else:
            disease_key_for_lookup = predicted_label # Use the full predicted label as the key
            display_disease_name = predicted_label.replace('_', ' ').strip() # Convert to human-readable

        # Extract crop_type from the predicted_label for display
        crop_type = predicted_label.split('_')[0]


        result = {
            "crop_type": crop_type,
            "disease_name": display_disease_name, # This is the human-readable name for UI
            "disease_key_for_lookup": disease_key_for_lookup, # This is the key for dictionary lookups
            "confidence": f"{confidence:.1f}%",
            "full_label": predicted_label # Keep original full label for reference
        }
        logging.info(f"diagnose_disease: Diagnosis Result: {result}")

        if confidence < 50:
            result["note"] = "Low confidence prediction. Please upload a clearer image for better results."

        return result

    except Exception as e:
        logging.error(f"Error during diagnosis: {e}", exc_info=True)
        return {"error": f"An error occurred during diagnosis: {str(e)}"}

def apply_image_filters(img_bytes, filters):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if 'brightness' in filters:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(float(filters['brightness']))
        if 'contrast' in filters:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(float(filters['contrast']))
        if 'saturation' in filters:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(float(filters['saturation']))
        if 'edge_enhance' in filters and float(filters['edge_enhance']) > 0:
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.0 + float(filters['edge_enhance']))
        if 'blur_reduce' in filters and float(filters['blur_reduce']) > 0:
            img = img.filter(ImageFilter.MedianFilter(size=3))
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.0 + float(filters['blur_reduce']))
        if 'color_balance' in filters:
            factor = float(filters['color_balance'])
            img_array = np.array(img)
            if factor > 0: # More red/yellow
                img_array[:,:,0] = np.clip(img_array[:,:,0] * (1 + factor), 0, 255) # Red
                img_array[:,:,1] = np.clip(img_array[:,:,1] * (1 + factor/2), 0, 255) # Green (less than red)
            else: # More blue
                img_array[:,:,2] = np.clip(img_array[:,:,2] * (1 - factor), 0, 255) # Blue
            img = Image.fromarray(img_array)
        if 'auto_contrast' in filters and filters['auto_contrast'] == 'true':
            img = ImageOps.autocontrast(img)
        if 'sharpen' in filters and filters['sharpen'] == 'true':
            img = img.filter(ImageFilter.SHARPEN)
        if 'grayscale' in filters and filters['grayscale'] == 'true':
            img = img.convert("L").convert("RGB")

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    except Exception as e:
        logging.error(f"Error applying image filters: {e}", exc_info=True)
        return None

# --- Application Initialization (Moved from @app.before_first_request) ---
# These functions will run once when the Flask application starts.
load_app_data()
if not load_models():
    logging.critical("Application failed to load ML models. Some features may not work.")

# --- Routes ---
@app.route('/')
def home():
    # Pass the current datetime object to the template
    return render_template('index.html', dark_mode=bool(app_data.get('dark_mode', False)), now=datetime.now())

@app.route('/toggle_theme', methods=['POST'])
def toggle_theme():
    app_data['dark_mode'] = not app_data.get('dark_mode', False) # Safely toggle
    save_app_data()
    return jsonify(success=True, dark_mode=app_data['dark_mode'])

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            try:
                # Read image data
                img_bytes = file.read()

                # Apply filters if provided
                filters = request.form.to_dict()
                if filters:
                    filtered_img_bytes = apply_image_filters(img_bytes, filters)
                    if filtered_img_bytes is None:
                        return jsonify({"error": "Failed to apply image filters."}), 500
                    img_bytes = filtered_img_bytes

                # Save the image temporarily for diagnosis and history (if it's a new upload)
                # Or, if it's a re-diagnosis, use the existing path
                original_filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                unique_filename = f"{timestamp}_{original_filename}"
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

                # Ensure the directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                with open(save_path, 'wb') as f:
                    f.write(img_bytes)

                diagnosis_result = diagnose_disease(img_bytes)

                if "error" in diagnosis_result:
                    logging.error(f"diagnose_page: Diagnosis returned an error: {diagnosis_result['error']}")
                    return jsonify(diagnosis_result), 500

                # Add to history
                history_entry = {
                    "image_path": save_path,
                    "diagnosis": diagnosis_result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                app_data['diagnosis_history'].insert(0, history_entry)
                if len(app_data['diagnosis_history']) > 50: # Limit history size
                    app_data['diagnosis_history'].pop()
                save_app_data()

                # Return the diagnosis result and the URL to the saved image
                image_url = url_for('uploaded_file', filename=unique_filename)
                return jsonify({
                    "success": True,
                    "diagnosis": diagnosis_result,
                    "image_url": image_url,
                    "image_filename": unique_filename # Send filename for report generation
                })
            except Exception as e:
                logging.error(f"Error during diagnose POST: {e}", exc_info=True)
                return jsonify({"error": f"Server error during diagnosis: {str(e)}"}), 500

    # GET request for diagnosis page
    return render_template(
        'diagnose.html',
        dark_mode=bool(app_data.get('dark_mode', False)),
        disease_info=disease_info, # Pass disease_info
        treatment_suggestions_data=treatment_suggestions_data, # Pass treatment_suggestions_data
        now=datetime.now(), # Pass the current datetime object to the template
        uploads_base_url=url_for('uploaded_file', filename='') # Pass the base URL for uploads
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the UPLOAD_FOLDER."""
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except FileNotFoundError:
        logging.error(f"File not found: {filename} in {app.config['UPLOAD_FOLDER']}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logging.error(f"Error serving uploaded file {filename}: {e}")
        return jsonify({"error": "Error serving file"}), 500


@app.route('/history')
def history_page():
    # Sort history by most recent by default
    sorted_history = sorted(app_data['diagnosis_history'], key=lambda x: x['timestamp'], reverse=True)
    # Pass the current datetime object to the template
    return render_template('history.html', history=sorted_history, dark_mode=bool(app_data.get('dark_mode', False)), class_labels=class_labels, now=datetime.now())

@app.route('/get_history_details/<path:filename>') # Changed to <path:filename>
def get_history_details(filename):
    # Ensure we are always working with just the basename
    # This handles cases where the client might send "uploads/filename.png" or just "filename.png"
    actual_filename = os.path.basename(filename)
    logging.info(f"get_history_details: Request for filename: {filename}, processed as: {actual_filename}")

    for item in app_data['diagnosis_history']:
        # Compare with the basename of the stored image_path
        stored_filename = os.path.basename(item.get('image_path', ''))
        if stored_filename == actual_filename:
            # Prepare data for display
            details = item['diagnosis']
            logging.info(f"get_history_details: Found history item. Details: {details}")
            
            # Use the 'disease_key_for_lookup' provided by the backend
            disease_lookup_key = details.get('disease_key_for_lookup')
            logging.info(f"get_history_details: Using disease_lookup_key: '{disease_lookup_key}'")
            
            # Attempt to get info using disease_lookup_key first
            info = disease_info.get(disease_lookup_key)
            
            # Final fallback to generic healthy info if still not found
            if info is None:
                logging.warning(f"get_history_details: Info still not found for '{disease_lookup_key}'. Falling back to generic healthy info.")
                info = disease_info.get("healthy")

            treatment_suggestions = treatment_suggestions_data.get(disease_lookup_key, treatment_suggestions_data.get("healthy"))
            
            logging.info(f"get_history_details: Fetched info: {info}")
            logging.info(f"get_history_details: Fetched treatment_suggestions: {treatment_suggestions}")

            return jsonify({
                "success": True,
                "details": details, # This still contains the human-readable 'disease_name'
                "info": info,
                "treatment_suggestions": treatment_suggestions,
                "image_url": url_for('uploaded_file', filename=actual_filename), # Use actual_filename here
                "timestamp": item['timestamp'] # Include timestamp for display
            })
    logging.warning(f"get_history_details: History item not found for filename: {filename}")
    return jsonify({"error": "History item not found"}), 404

@app.route('/delete_history_item/<path:filename>', methods=['POST']) # Changed to <path:filename>
def delete_history_item(filename):
    actual_filename = os.path.basename(filename) # Ensure we're working with basename
    initial_len = len(app_data['diagnosis_history'])
    app_data['diagnosis_history'] = [
        item for item in app_data['diagnosis_history']
        if not (item.get('image_path') and os.path.basename(item['image_path']) == actual_filename)
    ]
    if len(app_data['diagnosis_history']) < initial_len:
        # Also delete the associated image file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], actual_filename) # Use actual_filename here
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted image file: {file_path}")
        save_app_data()
        return jsonify(success=True)
    return jsonify(success=False, message="Item not found"), 404

@app.route('/clear_all_history', methods=['POST'])
def clear_all_history():
    # Delete all uploaded images first
    for item in app_data['diagnosis_history']:
        if item.get('image_path') and os.path.exists(item['image_path']):
            try:
                os.remove(item['image_path'])
            except Exception as e:
                logging.error(f"Error deleting history image file {item['image_path']}: {e}")
    app_data['diagnosis_history'] = []
    save_app_data()
    return jsonify(success=True)

@app.route('/export_all_history', methods=['GET'])
def export_all_history():
    if not app_data['diagnosis_history']:
        return jsonify({"error": "No history to export"}), 404

    export_data = []
    for item in app_data['diagnosis_history']:
        # Convert image_path to a URL for export if needed, or just keep path
        # For simplicity, let's keep the path as it is for local export
        export_data.append({
            "image_path": item.get("image_path"),
            "diagnosis": item.get("diagnosis"),
            "timestamp": item.get("timestamp")
        })

    temp_file_path = os.path.join(UPLOAD_FOLDER, f"crop_diagnosis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4)

    return send_file(temp_file_path, as_attachment=True, download_name=os.path.basename(temp_file_path))

@app.route('/import_history', methods=['POST'])
def import_history():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            imported_data = json.load(file)
            if not isinstance(imported_data, list):
                raise ValueError("Imported file does not contain a list of history records.")

            overwrite = request.form.get('overwrite') == 'true'
            if overwrite:
                app_data['diagnosis_history'].clear()

            for item in imported_data:
                if all(key in item for key in ["diagnosis", "timestamp"]):
                    # Validate image_path if it exists, and make it relative if needed
                    if 'image_path' in item and item['image_path']:
                        # For simplicity, we'll assume imported image paths are relative to UPLOAD_FOLDER
                        # or that the user will manually place them. If the path is absolute and doesn't exist,
                        # we'll nullify it.
                        if not os.path.exists(item['image_path']):
                            item['image_path'] = None # Mark as missing
                            logging.warning(f"Imported history item references missing image: {item.get('image_path', 'N/A')}")
                    app_data['diagnosis_history'].append(item)
                else:
                    logging.warning(f"Skipping invalid history item during import: {item}")

            save_app_data()
            return jsonify(success=True, message="History imported successfully.")

        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON file format: {e}"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logging.error(f"Error importing history: {e}", exc_info=True)
            return jsonify({"error": f"Failed to import history: {str(e)}"}), 500


@app.route('/feedback', methods=['GET', 'POST'])
def feedback_page():
    if request.method == 'POST':
        try:
            feedback_data = {
                "type": request.form.get("feedback_type"),
                "text": request.form.get("feedback_text"),
                "email": request.form.get("email"),
                "rating": request.form.get("rating"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0",
                "platform": "Web (Flask)"
            }

            if 'screenshot' in request.files and request.files['screenshot'].filename != '':
                screenshot_file = request.files['screenshot']
                filename = secure_filename(screenshot_file.filename)
                screenshots_dir = os.path.join(FEEDBACK_FOLDER, "screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)
                screenshot_path = os.path.join(screenshots_dir, filename)
                screenshot_file.save(screenshot_path)
                feedback_data["screenshot"] = filename

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = os.path.join(FEEDBACK_FOLDER, f"feedback_{timestamp}.json")
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2)

            return jsonify(success=True, message="Feedback submitted successfully!")
        except Exception as e:
            logging.error(f"Error submitting feedback: {e}", exc_info=True)
            return jsonify({"error": f"Failed to submit feedback: {str(e)}"}), 500
    # Pass the current datetime object to the template
    return render_template('feedback.html', dark_mode=bool(app_data.get('dark_mode', False)), now=datetime.now())

@app.route('/help')
def help_page():
    # Pass the current datetime object to the template
    return render_template('help.html', dark_mode=bool(app_data.get('dark_mode', False)), now=datetime.now())

@app.route('/about')
def about_page():
    # Pass the current datetime object to the template

    return render_template('about.html', dark_mode=bool(app_data.get('dark_mode', False)), now=datetime.now())

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    image_filename = data.get('image_filename')
    diagnosis_details = data.get('diagnosis_details')
    disease_info_data = data.get('disease_info')
    treatment_suggestions = data.get('treatment_suggestions')

    if not image_filename or not diagnosis_details:
        return jsonify({"error": "Missing data for report generation"}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    if not os.path.exists(image_path):
        return jsonify({"error": "Image file not found for report"}), 404

    temp_image_file_path = None # Initialize to None for finally block
    try:
        pdf = FPDF()
        pdf.set_font("Helvetica", "B", 16)
        pdf.add_page()

        pdf.cell(0, 10, "Crop Disease Diagnosis Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Image: {image_filename}", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Diagnosis Results:", ln=True)
        pdf.set_font("Helvetica", size=12)

        pdf.cell(0, 7, f"Crop Type: {diagnosis_details.get('crop_type', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Disease: {diagnosis_details.get('disease_name', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Confidence: {diagnosis_details.get('confidence', 'N/A')}", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Disease Information:", ln=True)
        pdf.set_font("Helvetica", size=12)
        if disease_info_data:
            pdf.multi_cell(0, 7, f"Description: {disease_info_data.get('description', 'N/A')}")
            pdf.ln(2)
            pdf.multi_cell(0, 7, f"Symptoms: {disease_info_data.get('symptoms', 'N/A')}")
        else:
            pdf.multi_cell(0, 7, "No detailed information available for this disease.")
        pdf.ln(10)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Treatment Suggestions:", ln=True)
        pdf.set_font("Helvetica", size=12)

        if treatment_suggestions:
            for suggestion in treatment_suggestions:
                pdf.multi_cell(0, 7, f"- {suggestion}")
        else:
            pdf.multi_cell(0, 7, "No specific treatment suggestions available for this diagnosis.")
        pdf.ln(10)

        # Embed image from the original file path
        try:
            # Open the image using PIL
            img_to_embed = Image.open(image_path).convert("RGB")
            img_width, img_height = img_to_embed.size
            max_pdf_width = 150
            max_pdf_height = 150

            if img_width > max_pdf_width or img_height > max_pdf_height:
                img_to_embed.thumbnail((max_pdf_width, max_pdf_height), Image.Resampling.LANCZOS)

            # Save the resized image to a temporary file for FPDF
            # PyFPDF 1.7.2 seems to prefer a file path over BytesIO for image embedding.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t_file:
                temp_image_file_path = t_file.name # Store path for finally block
                img_to_embed.save(temp_image_file_path, format="JPEG")
            
            logging.info(f"Attempting to embed image from temporary file: {temp_image_file_path}")

            pdf.ln(5)
            pdf.cell(0, 10, "Image Preview:", ln=True)
            x_pos = (pdf.w - img_to_embed.width) / 2
            # Pass the path of the temporary file to pdf.image
            pdf.image(temp_image_file_path, x=x_pos, w=img_to_embed.width, type="JPEG")
        except Exception as img_embed_e:
            logging.error(f"Failed to embed image in PDF: {img_embed_e}", exc_info=True)
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, "Error: Could not embed image in report.", ln=True, align="C")

        # Get PDF content as bytes
        pdf_content_bytes = pdf.output(dest='S') # Explicitly get as string
        # Ensure it's bytes for BytesIO, as fpdf.output() might return str in some environments
        pdf_output = io.BytesIO(pdf_content_bytes.encode('latin-1'))
        pdf_output.seek(0)

        filename = f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(pdf_output, as_attachment=True, download_name=filename, mimetype='application/pdf')

    except Exception as e:
        logging.error(f"Error generating PDF report: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500
    finally:
        # Clean up the temporary image file if it was created
        if temp_image_file_path and os.path.exists(temp_image_file_path):
            try:
                os.remove(temp_image_file_path)
                logging.info(f"Cleaned up temporary image file: {temp_image_file_path}")
            except Exception as e:
                logging.error(f"Error cleaning up temporary image file {temp_image_file_path}: {e}")

# --- Add this block to run the Flask app ---
if __name__ == '__main__':
    # You can specify host='0.0.0.0' to make it accessible from other devices on your network
    # and port=5000 (or any other port)
    app.run(debug=True, host='0.0.0.0', port=5000)