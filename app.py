import os
import io
import json
import logging
import requests 
import sqlite3
import numpy as np
import cv2
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from flask import Flask, render_template, request, jsonify, send_file, url_for, make_response, after_this_request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from fpdf import FPDF
import tempfile
from groq import Groq
from dotenv import load_dotenv
from flask import after_this_request
import secrets
import smtplib 
from email.mime.text import MIMEText 
from email.mime.multipart import MIMEMultipart
from flask_babel import Babel, _, gettext
from flask import session

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

app.secret_key = os.getenv('SECRET_KEY', 'dev_fallback_key')

# --- BABEL CONFIGURATION ---
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_SUPPORTED_LOCALES'] = ['en', 'hi', 'mr', 'kn'] # English, Hindi, Marathi & kannada 

babel = Babel(app)

def get_locale():
    # Check if user has a preference in session
    if 'language' in session:
        return session['language']
    # Otherwise try to match best language from browser request
    return request.accept_languages.best_match(app.config['BABEL_SUPPORTED_LOCALES'])

babel.init_app(app, locale_selector=get_locale)

# --- GROQ CONFIGURATION ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# 3. Check if the key exists (Safety Check)
if not groq_api_key:
    # This error will stop the app if you forgot the .env file
    raise ValueError("No API Key found! Make sure you created the .env file with GOOGLE_API_KEY inside.")

# 4. Configure GROQ
client = Groq(api_key=groq_api_key)

weather_api_key = os.getenv("WEATHER_API_KEY")

# 👇 SECURE CONFIGURATION 👇
SENDER_EMAIL = os.getenv("MAIL_USER")
SENDER_PASSWORD = os.getenv("MAIL_PASS")

# (If these are None, print a warning)
if not SENDER_EMAIL or not SENDER_PASSWORD:
    print("⚠️ WARNING: Email credentials not found in .env file!")

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
STATIC_FOLDER = 'static'
MODELS_FOLDER = 'models'
FEEDBACK_FOLDER = 'feedback' # Kept for screenshots
DB_NAME = 'crop_doctor.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

# --- Global Variables ---
disease_model = None
gatekeeper_model = None

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
    'Tomato_Spider_mites_Two_spotted_spider_mite': 33, 'Tomato_Target_Spot': 34,
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato_Tomato_mosaic_virus': 36,
    'Tomato_healthy': 37
}

# Create a reverse map (Number -> Name) ONCE at startup for speed
index_to_label = {v: k for k, v in class_labels.items()}

# --- Database Helper Functions ---
def get_db_connection():
    """Creates a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Access columns by name (row['id'])
    return conn

def init_db():
    """Initializes the database tables."""
    with get_db_connection() as conn:
        # History Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                crop_type TEXT,
                disease_name TEXT,
                confidence REAL,
                timestamp TEXT,
                full_details TEXT, -- Stores full JSON diagnosis
                disease_key_for_lookup TEXT
            )
        ''')
        # Feedback Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                text TEXT,
                email TEXT,
                rating INTEGER,
                screenshot_filename TEXT,
                timestamp TEXT
            )
        ''')
        # Settings Table (for Dark Mode)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        # --- NEW: Community/Expert Feed Table ---
        conn.execute('''
            CREATE TABLE IF NOT EXISTS community_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                crop_type TEXT,
                predicted_disease TEXT,
                confidence TEXT,
                user_question TEXT,
                expert_reply TEXT,
                reply_author TEXT,
                timestamp TEXT
            )
        ''')
        
        # 5. NEW: Expert Access Tokens Table (For Professional Keys)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS expert_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                assigned_to_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS verification_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                qualification TEXT NOT NULL,
                role TEXT DEFAULT 'Expert',
                status TEXT DEFAULT 'pending',
                timestamp TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS government_schemes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                ministry TEXT,
                description TEXT,
                eligible_state TEXT DEFAULT 'All',
                eligible_crop TEXT DEFAULT 'All',
                benefit TEXT,
                link TEXT,
                deadline DATE
            )
        ''')
        
        # --- NEW TABLE FOR IOT SENSORS ---
        conn.execute('''
            CREATE TABLE IF NOT EXISTS soil_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                moisture REAL,
                temperature REAL,
                humidity REAL,
                pump_status TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 1. NEW: Disease Heatmap Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS disease_heatmap (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_name TEXT NOT NULL,
                crop_type TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                city TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        schemes_data = [
            # --- ALL INDIA SCHEMES (Visible to everyone) ---
            ('Mission for Integrated Development of Horticulture (MIDH)', 'Ministry of Agriculture', 
            'Subsidy for planting material, greenhouses, and cold storage for fruit crops.', 
            'All', 'Apple, Grape, Cherry, Peach, Strawberry', '40-50% Subsidy on Infrastructure', 'https://midh.gov.in/', '2026-03-31'),
        
            ('Operation Greens (TOP Scheme)', 'Ministry of Food Processing', 
            'Price stabilization scheme specifically for Tomato, Onion, and Potato farmers.', 
            'All', 'Tomato, Potato', '50% Subsidy on Transport & Storage', 'https://www.mofpi.gov.in/', '2026-12-31'),
        
            ('Pradhan Mantri Fasal Bima Yojana (PMFBY)', 'Ministry of Agriculture', 
            'Insurance against crop loss due to pests (like Late Blight) or weather.', 
            'All', 'All', 'Insurance Claim Settlement', 'https://pmfby.gov.in/', '2026-07-31'),

            ('NFSM - Coarse Cereals (Maize)', 'Ministry of Agriculture', 
            'Support for improved seeds and technology demonstrations for Maize (Corn).', 
            'All', 'Corn', 'Free Hybrid Seeds & Field Demos', 'https://nfsm.gov.in/', '2026-06-30'),

            # --- NORTH INDIA SPECIFIC ---
            ('High Density Apple Plantation Scheme', 'Dept of Horticulture', 
            'Special subsidy for high-density apple orchards to boost production.', 
            'Jammu and Kashmir, Himachal Pradesh, Uttarakhand', 'Apple', '50% Subsidy on Plant Material', 'https://dirhortijmu.nic.in/', '2026-10-15'),

            ('Citrus Development Programme', 'National Horticulture Board', 
             'Rejuvenation of old orchards and new plantation support for Citrus fruits.', 
            'Punjab, Haryana, Rajasthan', 'Orange', 'Credit linked subsidy up to 40%', 'https://nhb.gov.in/', '2026-09-15'),

            # --- SOUTH INDIA SPECIFIC ---
            ('Drip Irrigation Subsidy (PMKSY)', 'Ministry of Jal Shakti', 
            'Massive subsidy for installing drip irrigation systems in water-scarce regions.', 
            'Tamil Nadu, Karnataka, Andhra Pradesh, Telangana', 'All', '75-100% Subsidy for Small Farmers', 'https://pmksy.gov.in/', '2026-05-20'),

            ('Coconut Development Board Scheme', 'Ministry of Agriculture', 
            'Assistance for expansion of area under coconut and replanting.', 
            'Kerala, Tamil Nadu, Karnataka, Goa', 'Coconut', '₹17,500 per hectare', 'https://www.coconutboard.gov.in/', '2026-08-10'),

            # --- CENTRAL & WEST INDIA ---
            ('NFSM - Oilseeds & Soybean', 'Ministry of Agriculture', 
            'Incentives for increasing oilseed production including Soybean.', 
            'Madhya Pradesh, Maharashtra, Gujarat', 'Soybean', '₹4000/quintal subsidy on seeds', 'https://nfsm.gov.in/', '2026-05-20'),

            ('Onion Storage Structure Scheme', 'Maharashtra Dept of Agriculture', 
            'Subsidy for constructing "Kanda Chawl" (Onion Storage) to prevent rotting.', 
            'Maharashtra, Gujarat', 'Onion', '₹87,500 per unit subsidy', 'https://krishi.maharashtra.gov.in/', '2026-04-01'),

            # --- EAST & NORTH-EAST INDIA ---
            ('BGREI - Rice & Veg Program', 'Ministry of Agriculture', 
            'Bringing Green Revolution to Eastern India - Tech support for rice and vegetables.', 
            'West Bengal, Bihar, Odisha, Assam', 'Rice, Tomato, Potato', 'Free Seeds & Tech Support', 'https://rkvy.nic.in/', '2026-06-15'),

            ('Organic Farming Mission (MOVCDNER)', 'Ministry of Agriculture', 
            'Promoting organic farming certification and value chains.', 
            'Sikkim, Assam, Arunachal Pradesh, Nagaland, Manipur, Mizoram, Tripura, Meghalaya', 'All', '₹50,000 per hectare for 3 years', 'https://pgsindia-ncof.gov.in/', '2026-11-30')
        ]
        
        conn.execute("DELETE FROM government_schemes")
        
        conn.executemany('''
            INSERT INTO government_schemes (name, ministry, description, eligible_state, eligible_crop, benefit, link, deadline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', schemes_data)
        
        # Initialize dark mode if not set
        conn.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', ('dark_mode', 'false'))
        conn.commit()
    logging.info("Database initialized successfully.")

def send_token_email(recipient_email, recipient_name, token, role):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = "✅ Your Expert Verification Approved!"

        body = f"""
        <html>
          <body>
            <h2 style="color:green;">Congratulations, {recipient_name}!</h2>
            <p>Your application to join as a <b>{role}</b> has been approved.</p>
            <p>Here is your unique Expert Token. Please keep it safe.</p>
            
            <div style="background:#f3f4f6; padding:15px; border-radius:10px; border-left: 5px solid green; margin: 20px 0;">
                <h3 style="margin:0; font-family:monospace; font-size:24px;">{token}</h3>
            </div>

            <p><b>How to use it:</b></p>
            <ol>
                <li>Go to the Community Page.</li>
                <li>Click "Reply" on a post.</li>
                <li>Paste this token in the Expert Token box.</li>
            </ol>
            <p>Thank you for helping our farmers! 🚜</p>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))

        # Connect to Gmail Server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"📧 Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

disease_info = {
    "Apple_Apple_scab": {
        "description": "Apple scab is a common fungal disease that affects apple and crabapple trees. It causes olive-green to brown spots on leaves, fruit, and twigs. Severe infections can lead to premature leaf drop and deformed fruit.",
        "symptoms": "Olive-green to brown spots on leaves, often with a velvety texture. Spots on fruit are dark, circular, and may become corky or cracked. Twig lesions can also occur.",
        "seven_day_plan": [
            {"day": 1, "task": "Sanitation: Rake and destroy all fallen leaves and fruit to remove the fungal source."},
            {"day": 2, "task": "Pruning: Prune the tree to open the canopy and improve air circulation (sunlight kills spores)."},
            {"day": 3, "task": "Fungicide: Apply a fungicide containing Captan, Sulfur, or Myclobutanil."},
            {"day": 4, "task": "Moisture Control: Avoid overhead irrigation; water only at the base of the tree."},
            {"day": 5, "task": "Monitor: Inspect fruit and younger leaves for new velvety olive-green spots."},
            {"day": 6, "task": "Nutrient Boost: Apply a foliar spray of seaweed extract to boost tree resistance."},
            {"day": 7, "task": "Re-Diagnose: Use CropDoctor AI to check if the disease spread has halted."}
        ]
    },
    "Apple_Black_rot": {
        "description": "Black rot is a fungal disease that affects apple trees, causing lesions on leaves, cankers on branches, and a characteristic black rot on fruit. It can lead to significant yield losses.",
        "symptoms": "Leaf spots are purplish with a brown center. Cankers on branches are sunken and discolored. Fruit rot begins as a brown spot that rapidly expands and turns black, often with concentric rings.",
        "seven_day_plan": [
            {"day": 1, "task": "Remove Mummies: Remove all dried, shriveled fruits (mummies) from the tree and ground."},
            {"day": 2, "task": "Prune Cankers: Prune out dead wood and cankers (infected bark) 4 inches below the visible rot."},
            {"day": 3, "task": "Fungicide: Apply a fungicide like Captan or Mancozeb to protect healthy tissue."},
            {"day": 4, "task": "Tool Cleaning: Sterilize pruning tools with bleach or alcohol after every cut."},
            {"day": 5, "task": "Observation: Check leaves for 'frog-eye' leaf spots which indicate active spread."},
            {"day": 6, "task": "Insect Control: Check for insects causing wounds on fruit (wounds allow fungus entry)."},
            {"day": 7, "task": "Re-Diagnose: Take a photo of the affected area to track healing or new spread."}
        ]
    },
    "Apple_Cedar_apple_rust": {
        "description": "Cedar-apple rust is a fungal disease that requires two hosts: apple/crabapple and cedar/juniper. It causes bright orange spots on apple leaves and can deform fruit.",
        "symptoms": "Bright orange-yellow spots on apple leaves, often with small black dots (spermagonia) in the center. Galls may form on cedar trees, producing gelatinous orange horns in wet weather.",
        "seven_day_plan": [
            {"day": 1, "task": "Identify Source: Look for Juniper/Cedar trees nearby (within 100-500 meters)."},
            {"day": 2, "task": "Remove Galls: If Cedars are yours, prune off the brown woody galls before they produce orange horns."},
            {"day": 3, "task": "Fungicide: Apply Myclobutanil or Sulfur-based fungicide to the *Apple* tree."},
            {"day": 4, "task": "Pruning: Remove heavily infected apple leaves to reduce spore load (if infection is light)."},
            {"day": 5, "task": "Observation: Check apple fruit bottoms for infection (can cause deformity)."},
            {"day": 6, "task": "General Care: Mulch around the base of the apple tree to retain moisture."},
            {"day": 7, "task": "Re-Diagnose: Check if new orange spots are appearing on fresh growth."}
        ]
    },
    "Blueberry_healthy": {
        "description": "The blueberry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy berry development.",
        "seven_day_plan": [
            {"day": 1, "task": "Soil Acid Check: Blueberries need acidic soil (pH 4.5-5.5). Test and amend if needed."},
            {"day": 3, "task": "Hydration: Ensure soil is consistently moist but well-drained."},
            {"day": 5, "task": "Pest Watch: Inspect for Spotted Wing Drosophila or birds eating berries."},
            {"day": 7, "task": "Mulching: Add pine bark or sawdust mulch to conserve acidity and moisture."}
        ]
    },
    "Cherry_Powdery_mildew": {
        "description": "Powdery mildew is a fungal disease that appears as white, powdery patches on the surface of leaves, stems, and sometimes fruit. Severe infections can stunt growth and reduce yield.",
        "symptoms": "White, powdery spots on leaves, stems, and fruit. Leaves may curl, distort, or turn yellow. Young leaves are often more susceptible.",
        "seven_day_plan": [
            {"day": 1, "task": "Wash Foliage: Spray leaves with water in the *morning* to wash off spores (mildew hates water)."},
            {"day": 2, "task": "Fungicide: Apply Sulfur dust, Potassium Bicarbonate, or Neem Oil."},
            {"day": 3, "task": "Pruning: Thin out the canopy to increase sunlight penetration (sunlight kills mildew)."},
            {"day": 4, "task": "Nitrogen Control: Avoid high-nitrogen fertilizers which promote susceptible soft growth."},
            {"day": 5, "task": "Monitor: Check young shoots for leaf curling or white powder."},
            {"day": 6, "task": "Clean Up: Remove fallen leaves from the base of the tree."},
            {"day": 7, "task": "Re-Diagnose: Verify that white patches are receding or turning gray (dying)."}
        ]
    },
    "Cherry_healthy": {
        "description": "The cherry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development.",
        "seven_day_plan": [
            {"day": 1, "task": "Routine Check: Look for aphids or borers on the trunk and stems."},
            {"day": 3, "task": "Watering: Deep water the tree root zone (avoid frequent shallow watering)."},
            {"day": 5, "task": "Bird Protection: Check bird netting if fruit is ripening."},
            {"day": 7, "task": "Weeding: Keep the base of the tree free of weeds and grass."}
        ]
    },
    "Corn_Cercospora_leaf_spot_Gray_leaf_spot": {
        "description": "Cercospora leaf spot (also known as Gray leaf spot) is a fungal disease of corn characterized by rectangular, gray-to-tan lesions on leaves. Severe infections can lead to significant yield loss.",
        "symptoms": "Long, narrow, rectangular lesions (1-2 inches long) that are gray to tan in color. Lesions are typically restricted by leaf veins. May appear water-soaked initially.",
        "seven_day_plan": [
            {"day": 1, "task": "Assessment: Estimate percentage of leaf area infected. If >50%, harvest may be impacted."},
            {"day": 2, "task": "Fungicide: Apply a fungicide with Azoxystrobin or Propiconazole if the crop is not yet mature."},
            {"day": 3, "task": "Residue Management: Plan to plow under crop debris after harvest (fungus overwinters there)."},
            {"day": 4, "task": "Observation: Check if lesions are expanding or merging."},
            {"day": 5, "task": "Rotation Plan: Mark this field for crop rotation (do not plant corn here next year)."},
            {"day": 6, "task": "Nutrients: Ensure Potassium levels are adequate (helps reduce disease severity)."},
            {"day": 7, "task": "Re-Diagnose: Monitor upper leaves to see if the disease is climbing the plant."}
        ]
    },
    "Corn_Common_rust": {
        "description": "Common rust is a fungal disease of corn characterized by the formation of reddish-brown pustules on leaves. Severe infections can reduce photosynthetic area and impact yield.",
        "symptoms": "Small, cinnamon-brown to reddish-brown pustules on both upper and lower leaf surfaces. Pustules may rupture, releasing powdery spores.",
        "seven_day_plan": [
            {"day": 1, "task": "Monitor Spread: Check if pustules are on upper leaves (ear leaf and above)."},
            {"day": 2, "task": "Fungicide: If infection is early and severe, apply Mancozeb or Pyraclostrobin."},
            {"day": 3, "task": "Variety Check: Note if this hybrid variety is susceptible for future planting decisions."},
            {"day": 4, "task": "Cool & Wet Watch: Rust spreads in cool/humid weather; monitor closely if rain is forecast."},
            {"day": 5, "task": "Nutrient Check: High nitrogen can increase severity; ensure balanced fertilization."},
            {"day": 6, "task": "Observation: Look for black pustules (teliospores) indicating the end of the cycle."},
            {"day": 7, "task": "Re-Diagnose: Check if new orange pustules are appearing on fresh leaves."}
        ]
    },
    "Corn_Northern_Leaf_Blight": {
        "description": "Northern Leaf Blight is a fungal disease of corn that causes long, elliptical, gray-green lesions on leaves. It can significantly reduce photosynthetic area and affect grain fill.",
        "symptoms": "Long, elliptical, gray-green to tan lesions on leaves, typically 1 to 6 inches long. Lesions may coalesce, blighting large areas of the leaf.",
        "seven_day_plan": [
            {"day": 1, "task": "Identification: Confirm cigar-shaped lesions. Distinguish from Gray Leaf Spot (rectangular)."},
            {"day": 2, "task": "Fungicide: Apply protection (Strobilurin/Triazole) if silking is occurring."},
            {"day": 3, "task": "Residue: Note that this fungus survives in debris; plan for tillage or rotation."},
            {"day": 4, "task": "Observation: Check lower leaves first, as infection usually moves upward."},
            {"day": 5, "task": "Weed Control: Remove grassy weeds that might host the pathogen."},
            {"day": 6, "task": "Harvest Plan: If severe, plan for early harvest to prevent stalk rot (secondary issue)."},
            {"day": 7, "task": "Re-Diagnose: Scan the 'Ear Leaf' (the most important leaf) for lesions."}
        ]
    },
    "Corn_healthy": {
        "description": "The corn plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy stalk and ear development.",
        "seven_day_plan": [
            {"day": 1, "task": "Scouting: Walk the field in a 'W' pattern to check for random issues."},
            {"day": 3, "task": "Watering: Corn needs significant water during silking/tasseling."},
            {"day": 5, "task": "Nutrients: Check for yellowing (Nitrogen deficiency) or purple leaves (Phosphorus deficiency)."},
            {"day": 7, "task": "Stem Check: Squeeze the lower stalk to check for firmness (stalk health)."}
        ]
    },
    "Grape_Black_rot": {
        "description": "Black rot is a destructive fungal disease of grapes that affects leaves, shoots, and fruit. It causes characteristic black, shriveled mummies on the fruit.",
        "symptoms": "Small, circular, reddish-brown spots on leaves that enlarge and turn tan with dark borders. Black, shriveled, raisin-like fruit mummies. Lesions on shoots and tendrils.",
        "seven_day_plan": [
            {"day": 1, "task": "Sanitation: Remove ALL shriveled, black 'mummy' berries from the vine and ground."},
            {"day": 2, "task": "Leaf Pruning: Remove infected leaves to improve air circulation around clusters."},
            {"day": 3, "task": "Fungicide: Apply Mancozeb, Captan, or Myclobutanil immediately."},
            {"day": 4, "task": "Weed Control: Clear tall weeds under vines that trap humidity."},
            {"day": 5, "task": "Monitor: Check healthy grape clusters for tiny brown spots (early infection)."},
            {"day": 6, "task": "Canopy Management: Tuck shoots to expose fruit to sunlight and wind."},
            {"day": 7, "task": "Re-Diagnose: Monitor existing spots; they should stop expanding if treatment works."}
        ]
    },
    "Grape_Esca_Black_Measles": {
        "description": "Esca (also known as Black Measles) is a complex of fungal diseases affecting grapevines, leading to wood decay and foliar symptoms. It can cause sudden vine collapse or chronic decline.",
        "symptoms": "Foliar symptoms include interveinal chlorosis (yellowing) followed by necrosis (browning), often with a 'tiger-stripe' pattern. Fruit may develop dark spots and shrivel. Wood symptoms include dark streaking and decay.",
        "seven_day_plan": [
            {"day": 1, "task": "Mark Vine: Tag the infected vine. Esca is chronic; the vine may need replacement eventually."},
            {"day": 2, "task": "Trunk Inspection: Check trunk for cracks or fungal conks. No cure for trunk rot."},
            {"day": 3, "task": "Pruning: Wait for dry weather, then prune out dead arms/canes."},
            {"day": 4, "task": "Wound Protection: Paint large pruning cuts with fungicidal paste to prevent new infection."},
            {"day": 5, "task": "Fruit Drop: Drop infected clusters to direct energy to vine survival."},
            {"day": 6, "task": "Hygiene: Disinfect shears between EVERY cut (Esca spreads via tools)."},
            {"day": 7, "task": "Re-Diagnose: Document symptoms for year-over-year comparison."}
        ]
    },
    "Grape_Leaf_blight": {
        "description": "Grape leaf blight is a general term for various conditions causing browning and death of leaf tissue. It can be caused by fungi, bacteria, or environmental factors.",
        "symptoms": "Irregular brown spots or patches on leaves, often starting at the margins or between veins. Affected areas may dry out and become brittle. Severe cases can lead to defoliation.",
        "seven_day_plan": [
            {"day": 1, "task": "Identify Cause: Check if spots have halos (fungal) or V-shapes (bacterial/environmental)."},
            {"day": 2, "task": "Water Stress Check: Ensure vines are not drought-stressed (can mimic blight)."},
            {"day": 3, "task": "Fungicide: Apply Copper-based spray (works for both fungal and some bacterial issues)."},
            {"day": 4, "task": "Pruning: Remove heavily blighted leaves to reduce spread."},
            {"day": 5, "task": "Nutrients: Check for Potassium deficiency (can cause marginal leaf burn)."},
            {"day": 6, "task": "Observation: Check new growth at the tips for symptoms."},
            {"day": 7, "task": "Re-Diagnose: Use the app to see if the blight has progressed."}
        ]
    },
    "Grape_healthy": {
        "description": "The grape plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy vine and fruit development.",
        "seven_day_plan": [
            {"day": 1, "task": "Canopy Management: Tuck stray shoots into trellis wires."},
            {"day": 3, "task": "Pest Scout: Look for leafhoppers or japanese beetles."},
            {"day": 5, "task": "Leaf Pulling: Remove leaves around fruit clusters to improve airflow (preventative)."},
            {"day": 7, "task": "General Check: Inspect the trunk base for borer damage."}
        ]
    },
    "Orange_Haunglongbing_Citrus_greening": {
        "description": "Huanglongbing (HLB), also known as Citrus Greening, is a devastating bacterial disease of citrus trees spread by psyllids. It causes yellowing of leaves, misshapen fruit, and eventual tree decline.",
        "symptoms": "Asymmetrical blotchy mottling or yellowing of leaves, often resembling nutrient deficiencies but not symmetrical. Small, lopsided, green-bottomed fruit that taste bitter. Premature fruit drop.",
        "seven_day_plan": [
            {"day": 1, "task": "Quarantine: This disease is fatal and incurable. Isolate the tree immediately."},
            {"day": 2, "task": "Vector Control: Spray surrounding trees with insecticide (Imidacloprid) to kill Asian Citrus Psyllids."},
            {"day": 3, "task": "Confirmation: Contact local agriculture extension for official lab testing if unsure."},
            {"day": 4, "task": "Removal: If confirmed, cut down the infected tree to save the rest of your orchard."},
            {"day": 5, "task": "Stump Treatment: Treat the stump with herbicide to prevent regrowth."},
            {"day": 6, "task": "Inspection: Check all nearby citrus trees for blotchy yellow leaves."},
            {"day": 7, "task": "Replacement: Plan to replant with certified disease-free nursery stock."}
        ]
    },
    "Peach_Bacterial_spot": {
        "description": "Bacterial spot is a common disease of peaches caused by Xanthomonas arboricola pv. pruni. It causes small, angular spots on leaves, lesions on twigs, and fruit spots.",
        "symptoms": "Small, angular, water-soaked spots on leaves that turn purplish-brown and may drop out, giving a 'shot-hole' appearance. Sunken, dark lesions on twigs and fruit. Fruit spots are dark, pitted, and may crack.",
        "seven_day_plan": [
            {"day": 1, "task": "Pruning: Remove twigs with 'spring cankers' (dark sunken lesions)."},
            {"day": 2, "task": "Bactericide: Spray with Copper fungicide (avoid high rates in hot weather to prevent leaf burn)."},
            {"day": 3, "task": "Fertilizer Management: Avoid excess Nitrogen (lush growth is more susceptible)."},
            {"day": 4, "task": "Windbreaks: Bacteria spread via wind-blown rain; consider planting windbreaks for future."},
            {"day": 5, "task": "Monitor: Check fruit for pitting or cracking."},
            {"day": 6, "task": "Ground Care: Maintain sod/grass between rows to reduce blowing sand/soil."},
            {"day": 7, "task": "Re-Diagnose: Monitor new leaves for 'shot-hole' symptoms."}
        ]
    },
    "Peach_healthy": {
        "description": "The peach plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development.",
        "seven_day_plan": [
            {"day": 1, "task": "Trunk Check: Look for gummy sap (borer damage) at the soil line."},
            {"day": 3, "task": "Thinning: If fruit set is heavy, thin peaches to 6-8 inches apart for size."},
            {"day": 5, "task": "Watering: Peaches need regular water during the final swell of fruit."},
            {"day": 7, "task": "Weeding: Keep the area under the canopy weed-free."}
        ]
    },
    "Pepper_bell_Bacterial_spot": {
        "description": "Bacterial spot is a common and destructive disease of pepper plants caused by Xanthomonas bacteria. It leads to dark, water-soaked spots on leaves and fruit.",
        "symptoms": "Small, dark, water-soaked spots on leaves that may develop yellow halos. Spots on fruit are dark, raised, and scabby. Severe infections can cause leaf yellowing and defoliation.",
        "seven_day_plan": [
            {"day": 1, "task": "Remove Infected: Remove heavily infected leaves and fallen fruit immediately."},
            {"day": 2, "task": "Spray: Apply Copper-based bactericide mixed with Mancozeb (improves efficacy)."},
            {"day": 3, "task": "Irrigation: STOP overhead watering. Water only at the soil level."},
            {"day": 4, "task": "Mulch: Apply straw or plastic mulch to prevent soil bacteria splashing onto leaves."},
            {"day": 5, "task": "Sanitize: Wash hands and tools before touching healthy plants."},
            {"day": 6, "task": "Observation: Check leaf undersides for water-soaked lesions."},
            {"day": 7, "task": "Re-Diagnose: Verify if leaf drop has slowed down."}
        ]
    },
    "Pepper_bell_healthy": {
        "description": "The bell pepper plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development.",
        "seven_day_plan": [
            {"day": 1, "task": "Support: Stake or cage plants as fruit gets heavy."},
            {"day": 3, "task": "Sunscald Prevention: Ensure enough leaf canopy covers the fruit from direct noon sun."},
            {"day": 5, "task": "Pest Scout: Look for aphids or thrips in the flowers."},
            {"day": 7, "task": "Harvest: Pick mature peppers to encourage new fruit set."}
        ]
    },
    "Potato_Early_blight": {
        "description": "Early blight is a fungal disease affecting potato and tomato plants. It causes dark, concentric ringed spots on older leaves, leading to defoliation and reduced yield.",
        "symptoms": "Dark brown to black spots with concentric rings (like a target) on older leaves. Lesions may also appear on stems and tubers. Yellowing of tissue around spots.",
        "seven_day_plan": [
            {"day": 1, "task": "Prune: Remove the lower 1/3rd of leaves if they are spotted/yellowing."},
            {"day": 2, "task": "Fungicide: Apply Chlorothalonil or Mancozeb."},
            {"day": 3, "task": "Stress Reduction: Ensure consistent watering (drought stress makes Early Blight worse)."},
            {"day": 4, "task": "Fertilize: Apply Nitrogen if the crop is young (weak plants get Early Blight)."},
            {"day": 5, "task": "Mulch: Mulch helps keep soil moisture stable and prevents spore splash."},
            {"day": 6, "task": "Monitor: Check stems for dark lesions."},
            {"day": 7, "task": "Re-Diagnose: Check if the 'bullseye' spots have stopped spreading."}
        ]
    },
    "Potato_Late_blight": {
        "description": "Late blight is a devastating disease of potato and tomato caused by a water mold (Phytophthora infestans). It can rapidly destroy entire crops under cool, wet conditions.",
        "symptoms": "Irregular, water-soaked, dark green to brown lesions on leaves, often starting at the leaf tips or edges. White, fuzzy fungal growth may be visible on the undersides of leaves during humid conditions. Brown, firm rot on tubers.",
        "seven_day_plan": [
            {"day": 1, "task": "EMERGENCY: Remove and bag/burn ALL infected plants immediately. Do not compost."},
            {"day": 2, "task": "Protect: Spray ALL nearby healthy plants with preventative fungicide (Mancozeb/Copper)."},
            {"day": 3, "task": "Inspect Tubers: Check exposed tubers near soil surface. Hilling up soil helps protect them."},
            {"day": 4, "task": "Dry Out: Improve drainage and airflow. Stop watering if soil is wet."},
            {"day": 5, "task": "Scout: Check field daily. Late blight can kill a field in 2-3 days."},
            {"day": 6, "task": "Harvest Plan: If widespread, kill vines (desiccate) 2 weeks before harvest to save tubers."},
            {"day": 7, "task": "Re-Diagnose: Any new fuzzy white growth means infection is still active."}
        ]
    },
    "Potato_healthy": {
        "description": "The potato plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy tuber development.",
        "seven_day_plan": [
            {"day": 1, "task": "Hilling: Mound soil around the base of the stems to cover developing tubers."},
            {"day": 3, "task": "Pest Scout: Look for Colorado Potato Beetles (orange with black stripes)."},
            {"day": 5, "task": "Watering: Potatoes need steady water. Uneven watering causes tuber cracks."},
            {"day": 7, "task": "Flower Check: Flowering indicates tubers are starting to size up."}
        ]
    },
    "Raspberry_healthy": {
        "description": "The raspberry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy berry development.",
        "seven_day_plan": [
            {"day": 1, "task": "Pruning: Remove weak or spindly canes to focus energy on fruiting canes."},
            {"day": 3, "task": "Trellising: Secure canes to wires to keep fruit off the ground."},
            {"day": 5, "task": "Watering: Raspberries have shallow roots; ensure topsoil remains moist."},
            {"day": 7, "task": "Harvest: Pick berries immediately when ripe to prevent pest attraction."}
        ]
    },
    "Soybean_healthy": {
        "description": "The soybean plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy pod development.",
        "seven_day_plan": [
            {"day": 1, "task": "Weeding: Soybeans compete poorly with weeds in early stages."},
            {"day": 3, "task": "Pest Scout: Check for Japanese beetles or stink bugs."},
            {"day": 5, "task": "Pod Check: Monitor pod filling stage. Water stress now reduces yield."},
            {"day": 7, "task": "Leaf Color: Pale green may indicate need for inoculation (Nitrogen fixing bacteria)."}
        ]
    },
    "Squash_Powdery_mildew": {
        "description": "Powdery mildew is a common fungal disease affecting squash and other cucurbits. It appears as white, powdery spots on leaves and stems, reducing photosynthesis and yield.",
        "symptoms": "White, powdery patches on the upper and lower surfaces of leaves and stems. Leaves may turn yellow, then brown, and eventually die. Reduced fruit size and quality.",
        "seven_day_plan": [
            {"day": 1, "task": "Remove: Cut off the most heavily infected (completely white) leaves."},
            {"day": 2, "task": "Spray: Apply Neem Oil, Sulfur, or a baking soda solution (1tbsp per gallon)."},
            {"day": 3, "task": "Airflow: Thin out dense vines to let air circulate."},
            {"day": 4, "task": "Sunlight: Mildew thrives in shade; ensure plants get full sun."},
            {"day": 5, "task": "Watering: Water the soil, not the leaves. Wash off spores if you must wet leaves."},
            {"day": 6, "task": "Monitor: Check undersides of leaves for new colonies."},
            {"day": 7, "task": "Re-Diagnose: Treatment should turn white powder to gray/brown (inactive)."}
        ]
    },
    "Strawberry_Leaf_scorch": {
        "description": "Leaf scorch is a fungal disease of strawberries causing purplish spots on leaves that eventually enlarge and coalesce, leading to a 'scorched' appearance.",
        "symptoms": "Small, purplish spots on leaves that enlarge into irregular, reddish-brown blotches. Margins of the spots may turn brown or reddish. Severe infections can cause leaves to dry up and curl.",
        "seven_day_plan": [
            {"day": 1, "task": "Sanitation: Remove and burn all dried/scorched leaves."},
            {"day": 2, "task": "Fungicide: Apply Captan or Copper fungicide."},
            {"day": 3, "task": "Irrigation: Use drip tape. Overhead sprinkling spreads this fungus rapidly."},
            {"day": 4, "task": "Weeding: Weeds trap moisture and restrict airflow; clear them out."},
            {"day": 5, "task": "Nutrients: Avoid excess Nitrogen (causes soft foliage)."},
            {"day": 6, "task": "Mulch: Ensure straw mulch is dry and not matting down wet."},
            {"day": 7, "task": "Re-Diagnose: Monitor new inner leaves for purple spots."}
        ]
    },
    "Strawberry_healthy": {
        "description": "The strawberry plant appears healthy. Continue with good plant care practices to maintain its health.",
        "symptoms": "Vibrant green leaves, no discoloration, spots, or wilting. Healthy fruit development.",
        "seven_day_plan": [
            {"day": 1, "task": "Runner Control: Clip off runners to focus energy on fruit (unless propagating)."},
            {"day": 3, "task": "Slug Watch: Check under mulch/straw for slugs damaging fruit."},
            {"day": 5, "task": "Harvest: Pick ripe fruit daily to prevent rot."},
            {"day": 7, "task": "Watering: Ensure 1 inch of water per week."}
        ]
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot is a common and destructive disease of tomato and pepper caused by several species of Xanthomonas bacteria. It leads to dark, water-soaked spots on leaves and fruit.",
        "symptoms": "Small, dark, water-soaked spots on leaves that may develop yellow halos. Spots on fruit are dark, raised, and scabby. Severe infections can cause leaf yellowing and defoliation.",
        "seven_day_plan": [
            {"day": 1, "task": "Dry Foliage: Stop overhead watering immediately. Bacteria swim in water films."},
            {"day": 2, "task": "Prune: Remove infected leaves (disinfect shears between every cut)."},
            {"day": 3, "task": "Spray: Apply Copper bactericide. (Antibiotics like Streptomycin are used in some commercial settings)."},
            {"day": 4, "task": "Stake: Get plants off the ground to improve drying."},
            {"day": 5, "task": "Mulch: Cover soil to prevent splash-back of soil bacteria."},
            {"day": 6, "task": "Monitor: Check fruit for 'scabby' raised spots."},
            {"day": 7, "task": "Re-Diagnose: Check if yellowing/spotting has slowed."}
        ]
    },
    "Tomato_Early_blight": {
        "description": "Early blight is a fungal disease affecting tomato and potato plants. It causes dark, concentric ringed spots on older leaves, leading to defoliation and reduced yield.",
        "symptoms": "Dark brown to black spots with concentric rings (like a target) on older leaves. Lesions may also appear on stems and fruit. Yellowing of tissue around spots.",
        "seven_day_plan": [
            {"day": 1, "task": "Isolate & Prune: Remove and burn all infected leaves immediately to stop spore spread."},
            {"day": 2, "task": "Fungicide Application: Spray Chlorothalonil or Copper-based fungicide in the evening."},
            {"day": 3, "task": "Water Management: Switch to drip irrigation. Do not water overhead to keep leaves dry."},
            {"day": 4, "task": "Monitor: Check for new lesions on middle leaves. Ensure soil moisture is moderate."},
            {"day": 5, "task": "Nutrient Boost: Apply a calcium-rich organic fertilizer to boost plant immunity."},
            {"day": 6, "task": "Sanitation: Clean all garden tools with a bleach solution."},
            {"day": 7, "task": "Re-Diagnose: Use CropDoctor AI to scan the plant again and check for recovery."}
        ]
    },
    "Tomato_Late_blight": {
        "description": "Late blight is a devastating disease of tomato and potato caused by a water mold (Phytophthora infestans). It can rapidly destroy entire crops under cool, wet conditions.",
        "symptoms": "Irregular, water-soaked, dark green to brown lesions on leaves, often starting at the leaf tips or edges. White, fuzzy fungal growth may be visible on the undersides of leaves during humid conditions. Brown, firm rot on fruit.",
        "seven_day_plan": [
            {"day": 1, "task": "Critical Action: Remove and destroy infected plants immediately. Do NOT compost."},
            {"day": 2, "task": "Prevention: Spray all remaining healthy tomato plants with Copper or Chlorothalonil."},
            {"day": 3, "task": "Inspection: Check stems for dark, greasy lesions (a sign of systemic infection)."},
            {"day": 4, "task": "Airflow: Aggressively prune healthy plants to maximize airflow and drying."},
            {"day": 5, "task": "Monitor Weather: Be alert during cool, rainy weather (high risk)."},
            {"day": 6, "task": "Fruit Check: Harvest mature green fruit early if the disease is spreading."},
            {"day": 7, "task": "Re-Diagnose: Any fuzzy white mold means the disease is still active."}
        ]
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold is a fungal disease of tomatoes, especially common in humid conditions. It causes velvety, olive-green to brown patches on the undersides of leaves.",
        "symptoms": "Yellowish spots on the upper leaf surface, with olive-green to brown, velvety fungal growth on the corresponding undersides. Leaves may curl, dry up, and fall off.",
        "seven_day_plan": [
            {"day": 1, "task": "Ventilation: If in a greenhouse, increase fans/venting immediately. Reduce humidity."},
            {"day": 2, "task": "Prune: Remove lower leaves and any leaves with dense mold underneath."},
            {"day": 3, "task": "Fungicide: Apply Copper or Chlorothalonil fungicide."},
            {"day": 4, "task": "Spacing: Ensure plants are not touching; prune side shoots (suckers)."},
            {"day": 5, "task": "Watering: Water only at the base, never on leaves."},
            {"day": 6, "task": "Temperature: Leaf mold hates heat; if possible, increase temp to >85°F to slow it."},
            {"day": 7, "task": "Re-Diagnose: Check undersides of new leaves for velvety growth."}
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot is a common fungal disease of tomatoes. It causes numerous small, circular spots on older leaves, often with dark borders and tiny black dots in the center.",
        "symptoms": "Numerous small, circular spots (1/8 to 1/4 inch) on older leaves. Spots have dark brown borders and tan to gray centers, often with tiny black specks (pycnidia) in the middle. Severe infections lead to defoliation.",
        "seven_day_plan": [
            {"day": 1, "task": "Sanitation: Remove lower leaves (splash zone). Septoria starts from the soil."},
            {"day": 2, "task": "Fungicide: Apply Chlorothalonil or Mancozeb sprays."},
            {"day": 3, "task": "Mulching: Apply a thick layer of mulch to bury fungal spores in the soil."},
            {"day": 4, "task": "Weed Control: Remove Nightshade weeds (related to tomatoes) nearby."},
            {"day": 5, "task": "Monitor: Check if spots are moving up the plant."},
            {"day": 6, "task": "Tools: Disinfect cages/stakes if reusing them."},
            {"day": 7, "task": "Re-Diagnose: New leaves should be free of small gray spots."}
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites are tiny pests that feed on plant cells, causing stippling and yellowing of leaves. Heavy infestations can lead to webbing and severe plant damage.",
        "symptoms": "Tiny yellow or white stippling (pinprick dots) on leaves. Yellowing, bronzing, or drying of leaves. Fine webbing on the undersides of leaves or between stems. Mites are barely visible to the naked eye.",
        "seven_day_plan": [
            {"day": 1, "task": "Isolate: Separate infected potted plants. Prune heavily webbed leaves."},
            {"day": 2, "task": "Water Blast: Spray undersides of leaves with a strong stream of water (mites hate water)."},
            {"day": 3, "task": "Treatment: Apply Insecticidal Soap or Neem Oil (coat undersides thoroughly)."},
            {"day": 4, "task": "Humidity: Mites love dry heat. Mist the leaves daily to increase humidity."},
            {"day": 5, "task": "Biocontrol: Consider releasing predatory mites (Phytoseiulus persimilis)."},
            {"day": 6, "task": "Monitor: Use a magnifying glass to check for moving specks."},
            {"day": 7, "task": "Re-Diagnose: Check if webbing has returned."}
        ]
    },
    "Tomato_Target_Spot": {
        "description": "Target spot is a fungal disease of tomatoes causing circular lesions with concentric rings, resembling a target. It affects leaves, stems, and fruit.",
        "symptoms": "Small, circular, water-soaked spots on leaves that enlarge and develop concentric rings, turning brown to black. A yellow halo may surround the spots. Lesions can also appear on stems and fruit.",
        "seven_day_plan": [
            {"day": 1, "task": "Airflow: Target spot loves humidity. Prune lower leaves to improve airflow."},
            {"day": 2, "task": "Fungicide: Apply Azoxystrobin or Copper fungicide."},
            {"day": 3, "task": "Inspection: Check fruit for firm, brown, sunken spots (indicates severe infection)."},
            {"day": 4, "task": "Nitrogen Check: Ensure plant nutrition is balanced (weak plants suffer more)."},
            {"day": 5, "task": "Weed: Remove weeds that block air circulation."},
            {"day": 6, "task": "Sanitation: Remove fallen fruit/leaves."},
            {"day": 7, "task": "Re-Diagnose: Monitor new growth for lesions."}
        ]
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease of tomatoes transmitted by whiteflies. It causes severe stunting and yellowing of leaves.",
        "symptoms": "Severe stunting of plants. Upward curling and yellowing of leaf margins, especially on younger leaves. Leaves become thickened and brittle. Flowers may drop, and fruit set is severely reduced.",
        "seven_day_plan": [
            {"day": 1, "task": "Immediate Removal: Bag and remove infected plants. There is NO CURE for the virus."},
            {"day": 2, "task": "Vector Control: Inspect remaining plants for Whiteflies (tiny white moths)."},
            {"day": 3, "task": "Treatment: Apply Neem Oil or Pyrethrin to control whiteflies on healthy plants."},
            {"day": 4, "task": "Weed: Remove weeds which may host asymptomatic virus."},
            {"day": 5, "task": "Traps: Set up yellow sticky traps to catch whiteflies."},
            {"day": 6, "task": "Reflective Mulch: Use silver mulch to repel whiteflies in future plantings."},
            {"day": 7, "task": "Re-Diagnose: Monitor neighbors for stunting or curling."}
        ]
    },
    "Tomato_Tomato_mosaic_virus": {
        "description": "Tomato Mosaic Virus (ToMV) is a highly contagious viral disease that causes mosaic patterns, mottling, and distortion of leaves in tomato plants.",
        "symptoms": "Light and dark green mosaic patterns or mottling on leaves. Leaves may be distorted, puckered, or fern-like. Stunting of plants and reduced fruit size or quality.",
        "seven_day_plan": [
            {"day": 1, "task": "Sanitation: Do not touch healthy plants after touching infected ones. Wash hands with milk or soap."},
            {"day": 2, "task": "Removal: Remove infected plants. Do not compost (virus survives high heat)."},
            {"day": 3, "task": "Tool Sterilization: Soak tools in a 10% bleach solution or trisodium phosphate."},
            {"day": 4, "task": "Smokers Warning: Tobacco mosaic virus is related; smokers should wash hands before gardening."},
            {"day": 5, "task": "Weed Control: Remove nightshade weeds."},
            {"day": 6, "task": "Soil Care: Virus persists in root debris; remove as much root material as possible."},
            {"day": 7, "task": "Re-Diagnose: Monitor neighbors for mottled leaves."}
        ]
    },
    "healthy": {
        "description": "The plant appears healthy with no visible signs of disease. Maintaining good cultural practices is key to preventing future issues.",
        "symptoms": "Uniform green coloration, no spots, lesions, or deformities. Leaves are turgid and vibrant.",
        "seven_day_plan": [
            {"day": 1, "task": "Routine Check: Inspect undersides of leaves for hidden pests (aphids/mites)."},
            {"day": 3, "task": "Soil Health: Verify soil moisture is adequate (not too wet, not too dry)."},
            {"day": 5, "task": "Weeding: Remove weeds around the base to prevent pest habitats."},
            {"day": 7, "task": "Maintenance: Continue standard watering and fertilization schedule."}
        ]
    }
}
 
treatment_suggestions_data = {
    "Apple_Apple_scab": [
        {"text": "Apply fungicides containing myclobutanil or sulfur", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Remove and destroy infected leaves and fruit", "link": ""},
        {"text": "Improve air circulation through pruning", "link": ""}
    ],
    "Apple_Black_rot": [
        {"text": "Prune out infected plant parts (canes, fruit)", "link": ""},
        {"text": "Apply fungicides like captan or mancozeb", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Ensure good air circulation and sunlight exposure", "link": ""}
    ],
    "Apple_Cedar_apple_rust": [
        {"text": "Remove nearby cedar trees (alternate host)", "link": ""},
        {"text": "Apply fungicides such as myclobutanil", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Plant resistant apple varieties", "link": ""}
    ],
    "Blueberry_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Cherry_Powdery_mildew": [
        {"text": "Apply sulfur or potassium bicarbonate", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Reduce humidity around plants", "link": ""},
        {"text": "Use resistant varieties when available", "link": ""},
        {"text": "Prune affected areas", "link": ""}
    ],
    "Cherry_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Corn_Cercospora_leaf_spot_Gray_leaf_spot": [
        {"text": "Apply fungicides containing chlorothalonil or copper", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Rotate crops to non-host plants", "link": ""},
        {"text": "Remove and destroy infected plant debris", "link": ""}
    ],
    "Corn_Common_rust": [
        {"text": "Use resistant varieties", "link": ""},
        {"text": "Apply fungicides if severe", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Practice good sanitation", "link": ""}
    ],
    "Corn_Northern_Leaf_Blight": [
        {"text": "Use resistant varieties", "link": ""},
        {"text": "Apply fungicides (Strobilurins or Triazoles)", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Practice crop rotation", "link": ""}
    ],
    "Corn_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Grape_Black_rot": [
        {"text": "Apply fungicides (Mancozeb, Myclobutanil, Captan)", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Sanitation: Remove and destroy all mummified berries", "link": ""},
        {"text": "Pruning: Prune to improve air circulation", "link": ""}
    ],
    "Grape_Esca_Black_Measles": [
        {"text": "Prune out diseased wood during dry periods", "link": ""},
        {"text": "Apply protective fungicidal paints to wounds", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Improve vineyard hygiene", "link": ""}
    ],
    "Grape_Leaf_blight": [
        {"text": "Apply fungicides (Copper-based or Mancozeb)", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Improve air circulation around plants", "link": ""},
        {"text": "Remove and dispose of infected leaves", "link": ""}
    ],
    "Grape_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Orange_Haunglongbing_Citrus_greening": [
        {"text": "No known cure; remove infected trees", "link": ""},
        {"text": "Control citrus psyllid vectors with Insecticides", "link": "https://www.bighaat.com/collections/insecticides"},
        {"text": "Plant certified disease-free nursery stock", "link": ""}
    ],
    "Peach_Bacterial_spot": [
        {"text": "Apply Copper-based bactericides", "link": "https://www.bighaat.com/collections/bactericides"},
        {"text": "Use disease-free seeds and transplants", "link": ""},
        {"text": "Avoid overhead watering", "link": ""},
        {"text": "Practice crop rotation", "link": ""}
    ],
    "Peach_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Pepper_bell_Bacterial_spot": [
        {"text": "Apply Copper-based bactericides", "link": "https://www.bighaat.com/collections/bactericides"},
        {"text": "Use disease-free seeds and transplants", "link": ""},
        {"text": "Avoid overhead watering", "link": ""},
        {"text": "Practice crop rotation", "link": ""}
    ],
    "Pepper_bell_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Potato_Early_blight": [
        {"text": "Apply Copper-based fungicides", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Practice crop rotation", "link": ""},
        {"text": "Remove and destroy infected plant material", "link": ""}
    ],
    "Potato_Late_blight": [
        {"text": "Apply fungicides containing Chlorothalonil or Copper", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Destroy infected plants immediately", "link": ""},
        {"text": "Avoid overhead watering", "link": ""}
    ],
    "Potato_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Raspberry_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Soybean_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Squash_Powdery_mildew": [
        {"text": "Apply Sulfur or Potassium Bicarbonate", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Reduce humidity around plants", "link": ""},
        {"text": "Use resistant varieties when available", "link": ""},
        {"text": "Prune affected areas", "link": ""}
    ],
    "Strawberry_Leaf_scorch": [
        {"text": "Apply fungicides with Captan or Myclobutanil", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Sanitation: Remove and destroy infected leaves", "link": ""},
        {"text": "Mulching: Use mulch to prevent splashing spores", "link": ""}
    ],
    "Strawberry_healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Tomato_Bacterial_spot": [
        {"text": "Apply Copper-based bactericides", "link": "https://www.bighaat.com/collections/bactericides"},
        {"text": "Use disease-free seeds and transplants", "link": ""},
        {"text": "Avoid overhead watering", "link": ""},
        {"text": "Practice crop rotation", "link": ""}
    ],
    "Tomato_Early_blight": [
        {"text": "Apply Copper-based fungicides", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Practice crop rotation", "link": ""},
        {"text": "Remove and destroy infected plant material", "link": ""}
    ],
    "Tomato_Late_blight": [
        {"text": "Apply fungicides containing Chlorothalonil or Copper", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Destroy infected plants immediately", "link": ""},
        {"text": "Avoid overhead watering", "link": ""}
    ],
    "Tomato_Leaf_Mold": [
        {"text": "Improve air circulation", "link": ""},
        {"text": "Reduce humidity in greenhouses", "link": ""},
        {"text": "Apply fungicides (Chlorothalonil or Mancozeb)", "link": "https://www.bighaat.com/collections/fungicides"}
    ],
    "Tomato_Septoria_leaf_spot": [
        {"text": "Apply fungicides with Chlorothalonil or Mancozeb", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Remove infected leaves and plant debris", "link": ""},
        {"text": "Avoid overhead watering", "link": ""}
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        {"text": "Apply Insecticidal Soaps or Neem Oil", "link": "https://www.bighaat.com/collections/insecticides"},
        {"text": "Increase humidity around plants (mist foliage)", "link": ""},
        {"text": "Introduce predatory mites", "link": ""}
    ],
    "Tomato_Target_Spot": [
        {"text": "Apply fungicides containing Chlorothalonil or Mancozeb", "link": "https://www.bighaat.com/collections/fungicides"},
        {"text": "Practice crop rotation and sanitation", "link": ""},
        {"text": "Ensure good air circulation", "link": ""}
    ],
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": [
        {"text": "No cure; manage Whitefly vectors with Insecticides", "link": "https://www.bighaat.com/collections/insecticides"},
        {"text": "Use resistant varieties if available", "link": ""},
        {"text": "Remove infected plants immediately", "link": ""}
    ],
    "Tomato_Tomato_mosaic_virus": [
        {"text": "No chemical cure; remove infected plants", "link": ""},
        {"text": "Disinfect tools and hands", "link": ""},
        {"text": "Use resistant varieties", "link": ""}
    ],
    "healthy": [
        {"text": "Continue regular monitoring", "link": ""},
        {"text": "Maintain good cultural practices", "link": "https://www.bighaat.com/collections/growth-promoters"},
        {"text": "Preventative measures", "link": ""}
    ]
}

# Populate generic fallback for all specific healthy keys if missing
if "healthy" in treatment_suggestions_data:
    for key in disease_info:
        if "healthy" in key.lower() and key not in treatment_suggestions_data:
            # Use .copy() so they are independent lists
            treatment_suggestions_data[key] = treatment_suggestions_data["healthy"].copy()
            
# --- ML Functions ---
# [Replace the existing def load_models() function with this]
def load_models():
    global disease_model, gatekeeper_model
    
    # 1. Load Main Disease Model
    disease_model_path = os.path.join(MODELS_FOLDER, "my_model.keras")
    try:
        if os.path.exists(disease_model_path):
            disease_model = load_model(disease_model_path)
            # Warmup
            disease_model.predict(np.zeros((1, 150, 150, 3)), verbose=0)
            logging.info("✅ Disease Model Loaded.")
        else:
            logging.error(f"❌ Disease Model missing at {disease_model_path}")
            return False

        # 2. Load Gatekeeper Model (NEW)
        gatekeeper_path = os.path.join(MODELS_FOLDER, "gatekeeper_model.keras")
        if os.path.exists(gatekeeper_path):
            gatekeeper_model = load_model(gatekeeper_path)
            # Warmup (MobileNetV2 uses 224x224 input)
            gatekeeper_model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
            logging.info("✅ Gatekeeper AI Loaded.")
        else:
            logging.warning("⚠️ Gatekeeper AI missing. Validation will be disabled.")
            
        return True
    except Exception as e:
        logging.critical(f"Failed to load models: {e}")
        return False

def diagnose_disease(img_bytes):
    if not disease_model: return {"error": "Model not loaded."}
    
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((150, 150), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = disease_model.predict(img_array, verbose=0)
        idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        predicted_label = index_to_label[idx]

        disease_key = "healthy" if "healthy" in predicted_label.lower() else predicted_label
        display_name = "Healthy" if "healthy" in predicted_label.lower() else predicted_label.replace('_', ' ')

        return {
            "crop_type": predicted_label.split('_')[0],
            "disease_name": display_name,
            "disease_key_for_lookup": predicted_label, # Use full label for precise lookup
            "confidence": f"{confidence:.1f}%",
            "note": "Low confidence." if confidence < 50 else ""
        }
    except Exception as e:
        logging.error(f"Diagnosis error: {e}")
        return {"error": str(e)}

def apply_image_filters(img_bytes, filters):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if 'brightness' in filters: img = ImageEnhance.Brightness(img).enhance(float(filters['brightness']))
        if 'contrast' in filters: img = ImageEnhance.Contrast(img).enhance(float(filters['contrast']))
        if 'saturation' in filters: img = ImageEnhance.Color(img).enhance(float(filters['saturation']))
        if 'auto_contrast' in filters: img = ImageOps.autocontrast(img)
        if 'grayscale' in filters: img = img.convert("L").convert("RGB")
        
        output = io.BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
    except Exception as e:
        logging.error(f"Filter error: {e}")
        return None
    

# [Paste this where the old is_plant_image function was]
def verify_plant_ai(img_bytes):
    """
    Returns True if AI is >60% sure it's a plant.
    """
    if not gatekeeper_model:
        return True # Fallback: if model missing, allow everything
        
    try:
        # 1. Preprocess for MobileNetV2 (224x224)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 2. Predict
        prediction = gatekeeper_model.predict(img_array, verbose=0)[0][0]
        
        # 3. STRICT Threshold Logic (0.60 = 60%)
        # If prediction > 0.60, it is a Plant (1).
        is_plant = prediction > 0.60
        
        status = "✅ ACCEPTED" if is_plant else "❌ REJECTED"
        confidence = prediction * 100
        logging.info(f"Gatekeeper Check: {confidence:.2f}% ({status})")
        
        return is_plant
        
    except Exception as e:
        logging.error(f"Gatekeeper Check Failed: {e}")
        return True # Fail open to avoid blocking users on error

# --- Init ---
init_db()
if not load_models(): logging.warning("Model load failed.")

def get_dark_mode():
    with get_db_connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='dark_mode'").fetchone()
        return row['value'] == 'true' if row else False

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html', dark_mode=get_dark_mode(), now=datetime.now())

@app.route('/toggle_theme', methods=['POST'])
def toggle_theme():
    current = get_dark_mode()
    new_val = 'false' if current else 'true'
    with get_db_connection() as conn:
        conn.execute("UPDATE settings SET value = ? WHERE key = 'dark_mode'", (new_val,))
        conn.commit()
    return jsonify(success=True, dark_mode=(new_val == 'true'))

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose_page():
    if request.method == 'POST':
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        if not file.filename: return jsonify({"error": "No filename"}), 400

        try:
            img_bytes = file.read()
            filters = request.form.to_dict()
            if filters:
                processed = apply_image_filters(img_bytes, filters)
                if processed: img_bytes = processed

            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(save_path, 'wb') as f: f.write(img_bytes)
            
            # --- NEW AI CHECK ---
            if not verify_plant_ai(img_bytes):
                os.remove(save_path) # Delete the junk file
                return jsonify({
                    'error': 'Invalid Image Detected',
                    'message': 'Our AI Analysis indicates this is likely NOT a crop leaf. Please upload a clear photo of a plant.'
                }), 400

            result = diagnose_disease(img_bytes)
            if "error" in result: return jsonify(result), 500
            
            # --- NEW CODE START: Fetch 7-Day Plan ---
            lookup_key = result.get('disease_key_for_lookup', 'healthy')
            
            # Get specific plan or fallback to healthy plan
            disease_data = disease_info.get(lookup_key, disease_info['healthy'])
            seven_day_plan = disease_data.get('seven_day_plan', disease_info['healthy']['seven_day_plan'])
            
            # Capture Location Data from the Frontend Form
            lat = request.form.get('latitude')
            long = request.form.get('longitude')
            city = request.form.get('city')

            # Save to SQLite
            with get_db_connection() as conn:
                # [EXISTING] Save to History
                conn.execute('''
                    INSERT INTO history (image_filename, crop_type, disease_name, confidence, timestamp, full_details, disease_key_for_lookup)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    filename,
                    result['crop_type'],
                    result['disease_name'],
                    float(result['confidence'].strip('%')),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    json.dumps(result),
                    result['disease_key_for_lookup']
                ))
                # Save to Heatmap (Only if location exists and it's a disease)
                if lat and long and "healthy" not in result['disease_name'].lower():
                    try:
                        conn.execute('''
                            INSERT INTO disease_heatmap (disease_name, crop_type, latitude, longitude, city, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            result['disease_name'], 
                            result['crop_type'], 
                            float(lat), 
                            float(long), 
                            city, 
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ))
                    except Exception as e:
                        logging.error(f"Failed to save heatmap data: {e}")
                
                conn.commit()

            return jsonify({
                "success": True,
                "diagnosis": result,
                "seven_day_plan": seven_day_plan,
                "image_url": url_for('uploaded_file', filename=filename),
                "image_filename": filename
            })
        except Exception as e:
            logging.error(f"Diagnose error: {e}")
            return jsonify({"error": str(e)}), 500

    return render_template('diagnose.html', 
                           dark_mode=get_dark_mode(), 
                           disease_info=disease_info, 
                           treatment_suggestions_data=treatment_suggestions_data,
                           now=datetime.now(),
                           uploads_base_url=url_for('uploaded_file', filename=''))


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    # 1. Construct the full path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 2. Check if the file actually exists
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        # 3. If missing, return a 404 error instead of crashing the server
        return "Image not found", 404

# Add this route to serve the service worker from the ROOT
@app.route('/service-worker.js')
def service_worker():
    return send_file(os.path.join(app.static_folder, 'service-worker.js'), mimetype='application/javascript')

@app.route('/history')
def history_page():
    sort_by = request.args.get('sort_by', 'recent')
    filter_crop = request.args.get('filter_crop', 'all')

    query = "SELECT * FROM history"
    params = []
    
    if filter_crop != 'all':
        query += " WHERE lower(crop_type) = ?"
        params.append(filter_crop.lower())

    if sort_by == 'oldest': query += " ORDER BY timestamp ASC"
    elif sort_by == 'highest_confidence': query += " ORDER BY confidence DESC"
    elif sort_by == 'lowest_confidence': query += " ORDER BY confidence ASC"
    else: query += " ORDER BY timestamp DESC"

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    history_data = []
    for row in rows:
        diag_details = json.loads(row['full_details'])
        lookup_key = diag_details.get('disease_key_for_lookup', 'healthy')
        info = disease_info.get(lookup_key, disease_info['healthy'])
        treatments = treatment_suggestions_data.get(lookup_key, treatment_suggestions_data['healthy'])
        
        # 4. Bundle everything together
        history_data.append({
            "image_path": row['image_filename'], # Just the filename
            "diagnosis": diag_details,
            "timestamp": row['timestamp'],
            "info": info,                   # <--- ADDED THIS
            "treatment_suggestions": treatments # <--- ADDED THIS
        })

    return render_template('history.html', 
                           history=history_data, 
                           dark_mode=get_dark_mode(), 
                           class_labels=class_labels, 
                           now=datetime.now())

@app.route('/get_history_details/<path:filename>')
def get_history_details(filename):
    actual_filename = os.path.basename(filename)
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM history WHERE image_filename = ?", (actual_filename,)).fetchone()
    
    if row:
        details = json.loads(row['full_details'])
        key = row['disease_key_for_lookup']
        info = disease_info.get(key) or disease_info.get('healthy')
        treatments = treatment_suggestions_data.get(key) or treatment_suggestions_data.get('healthy') or []
        
        return jsonify({
            "success": True, 
            "details": details, 
            "info": info, 
            "treatment_suggestions": treatments,
            "image_url": url_for('uploaded_file', filename=actual_filename),
            "timestamp": row['timestamp']
        })
    return jsonify({"error": "Not found"}), 404

@app.route('/delete_history_item/<path:filename>', methods=['POST'])
def delete_history_item(filename):
    actual_filename = os.path.basename(filename)
    
    with get_db_connection() as conn:
        conn.execute("DELETE FROM history WHERE image_filename = ?", (actual_filename,))
        conn.commit()
    
    # --- COMMENTED OUT: Stop deleting the file so Community posts stay safe ---
    # path = os.path.join(app.config['UPLOAD_FOLDER'], actual_filename)
    # if os.path.exists(path): os.remove(path)
    # ------------------------------------------------------------------------

    return jsonify(success=True)

@app.route('/clear_all_history', methods=['POST'])
def clear_all_history():
    with get_db_connection() as conn:
        # We don't need to fetch rows anymore since we aren't deleting files
        # rows = conn.execute("SELECT image_filename FROM history").fetchall()
        
        # Only delete the data from the database
        conn.execute("DELETE FROM history")
        conn.commit()
    
    # --- COMMENTED OUT TO SAVE IMAGES FOR BACKUP/RESTORE ---
    # for row in rows:
    #     path = os.path.join(app.config['UPLOAD_FOLDER'], row['image_filename'])
    #     if os.path.exists(path): os.remove(path)
    # -------------------------------------------------------

    return jsonify(success=True)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback_page():
    if request.method == 'POST':
        try:
            screenshot_filename = None
            if 'screenshot' in request.files:
                f = request.files['screenshot']
                if f.filename:
                    screenshot_filename = secure_filename(f.filename)
                    f.save(os.path.join(FEEDBACK_FOLDER, screenshot_filename))

            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO feedback (type, text, email, rating, screenshot_filename, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    request.form.get("feedback_type"),
                    request.form.get("feedback_text"),
                    request.form.get("email"),
                    request.form.get("rating"),
                    screenshot_filename,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                conn.commit()
            return jsonify(success=True)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return render_template('feedback.html', dark_mode=get_dark_mode(), now=datetime.now())

@app.route('/export_all_history', methods=['GET'])
def export_all_history():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM history").fetchall()
    
    data = []
    for row in rows:
        data.append({
            "diagnosis": json.loads(row['full_details']),
            "timestamp": row['timestamp'],
            # IMPROVEMENT 1: Use 'image_filename' to match your DB and Frontend
            "image_filename": row['image_filename'] 
        })
    
    # Create the file
    filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    # IMPROVEMENT 2: Delete the file from the server after sending it
    @after_this_request
    def remove_file(response):
        try:
            os.remove(path)
        except Exception as e:
            logging.error(f"Error removing export file: {e}")
        return response

    return send_file(path, as_attachment=True, download_name=filename, mimetype='application/json')

@app.route('/import_history', methods=['POST'])
def import_history():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    overwrite = request.form.get('overwrite') == 'true'
    
    try:
        data = json.load(file)
        with get_db_connection() as conn:
            if overwrite: conn.execute("DELETE FROM history")
            for item in data:
                diag = item['diagnosis']
                
                # IMPROVED LOGIC: Try 'image_filename' first (new standard), 
                # then fallback to 'image_path' (old exports), 
                # finally fallback to 'unknown.png'.
                img_name = item.get('image_filename', item.get('image_path', 'unknown.png'))

                conn.execute('''
                    INSERT INTO history (image_filename, crop_type, disease_name, confidence, timestamp, full_details, disease_key_for_lookup)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    img_name,
                    diag['crop_type'],
                    diag['disease_name'],
                    float(str(diag['confidence']).strip('%')), # Added str() safety just in case
                    item['timestamp'],
                    json.dumps(diag),
                    diag.get('disease_key_for_lookup', diag['disease_name'])
                ))
            conn.commit()
        return jsonify(success=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/help')
def help_page():
    return render_template('help.html', dark_mode=get_dark_mode(), now=datetime.now())

@app.route('/about')
def about_page():
    return render_template('about.html', dark_mode=get_dark_mode(), now=datetime.now())

# --- DISEASE LIBRARY ROUTE ---
@app.route('/library')
def library_page():
    # We pass the 'disease_info' dictionary to the template
    return render_template('library.html', diseases=disease_info, now=datetime.now())

# --- LEGAL ROUTES ---
@app.route('/privacy')
def privacy_page():
    return render_template('privacy.html', now=datetime.now())

@app.route('/terms')
def terms_page():
    return render_template('terms.html', now=datetime.now())

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    filename = data.get('image_filename')
    details = data.get('diagnosis_details')
    info = data.get('disease_info')
    treatments = data.get('treatment_suggestions')
    seven_day_plan = data.get('seven_day_plan', [])

    if not filename or not details: return jsonify({"error": "Missing data"}), 400
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 15, "Crop Disease Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"Disease Detected: {details['disease_name']}", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Confidence: {details['confidence']}", ln=True)
    pdf.cell(0,8, f"Crop Type: {details['crop_type']}", ln=True)
    pdf.ln(5)
    
    if os.path.exists(image_path):
        try:
            tmp_filename = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_filename = tmp.name
                Image.open(image_path).convert('RGB').save(tmp, format='JPEG')
            
            if tmp_filename:
                # Calculate center position: (A4 width 210mm - Image width 80mm) / 2 = 65
                pdf.image(tmp_filename, x=65, y=None, w=80) 
                pdf.ln(5) # Add space after image
                try: os.unlink(tmp_filename)
                except Exception: pass
        except Exception as e:
            logging.error(f"PDF Image error: {e}")
    
    if info:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(240, 240, 240) # Light Gray Background
        pdf.cell(0, 8, "Symptoms & Description:", ln=True, fill=True)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 6, info.get('symptoms', 'N/A'))
        
    pdf.ln(5)
    
    if treatments:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 100, 0) # Green Header
        pdf.cell(0, 10, "Recommended Actions & Treatments:", ln=True)
        pdf.set_text_color(0, 0, 0) # Reset Black
        pdf.set_font("Helvetica", size=12)
        for t in treatments:
            if isinstance(t, dict):
                text_content = t.get('text', '')
                link_content = t.get('link', '')
                
                # Print the main text
                pdf.cell(0, 7, f"- {text_content}", ln=True)
                
                # If there is a link, add a blue clickable line below it
                if link_content:
                    pdf.set_text_color(0, 0, 255) # Blue
                    pdf.set_font("Helvetica", "I", 10) # Italic Small
                    pdf.cell(0, 5, "   [Click to View Product / Buy Now]", ln=True, link=link_content)
                    pdf.set_text_color(0, 0, 0) # Reset Black
                    pdf.set_font("Helvetica", size=12) # Reset Normal
            else:
                # Fallback for old simple string data
                pdf.cell(0, 7, f"- {t}", ln=True) 
    
    if seven_day_plan:
        pdf.ln(5) # Add some spacing
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_text_color(0, 100, 0) # Green Header
        pdf.cell(0, 10, "Structured 7-Day Recovery Plan:", ln=True)
        
        pdf.set_font("Helvetica", '', 11)
        pdf.set_text_color(0, 0, 0) # Reset to Black
        
        for item in seven_day_plan:
            day_num = item.get('day', '?')
            task_desc = item.get('task', '')
            
            # Format: "Day 1: Remove leaves..."
            line_text = f"Day {day_num}: {task_desc}"
            
            # Use multi_cell to handle text wrapping automatically
            pdf.multi_cell(0, 7, line_text)
            pdf.ln(1) # Small gap between days
                
    pdf_out = pdf.output(dest='S').encode('latin-1')
    return send_file(io.BytesIO(pdf_out), download_name="report.pdf", as_attachment=True, mimetype='application/pdf')

@app.route('/chat', methods=['POST'])
def chat_with_dr_crop():
    data = request.json
    user_message = data.get('message')
    disease_key = data.get('disease_key')
    
    lang_code = session.get('language', 'en')
    
    # Map code to the full English name for the prompt
    lang_map = {
        'en': 'English',
        'hi': 'Hindi',
        'mr': 'Marathi',
        'kn': 'Kannada'
    }
    # e.g., "Hindi", "Marathi", or "English"
    target_language = lang_map.get(lang_code, 'English')
    
    
    # --- 1. PREPARE CONTEXT (Same Logic as before) ---
    
    # Extract Crop Name (e.g., "Potato" from "Potato_Early_blight")
    # If disease_key is None/Empty, default to "this crop"
    crop_name = disease_key.split('_')[0] if disease_key and '_' in disease_key else "this crop"

    # RAG: Retrieve context from your existing dictionaries
    # We use .get() to avoid crashing if the key is missing
    context_info = disease_info.get(disease_key, {}) if disease_key else disease_info.get('healthy', {})
    treatment_info = treatment_suggestions_data.get(disease_key, [])
    
    # ✅ FIX: Handle both strings and dictionaries (objects with links)
    if treatment_info:
        treatment_list = []
        for t in treatment_info:
            if isinstance(t, dict):
                # If it's a dictionary, extract the 'text' field
                treatment_list.append(t.get('text', ''))
            else:
                # If it's already a string, just add it
                treatment_list.append(str(t))
        treatment_str = ", ".join(treatment_list)
    else:
        treatment_str = "No specific treatments found."
    
    # --- 2. BUILD THE SYSTEM PROMPT (The "Brain") ---
    # This string defines the 3 personalities (Pathologist, Economist, Agronomist)
    system_prompt = f"""
    You are 'Dr. Crop', an expert AI consultant for Agriculture. You have Three distinct roles:
    
    ROLE 1: PLANT PATHOLOGIST (Disease Doctor)
    - The user has a {crop_name} plant.
    - Diagnosis: {disease_key.replace('_', ' ') if disease_key else 'Unknown'}
    - Description: {context_info.get('description', 'N/A')}
    - Symptoms: {context_info.get('symptoms', 'N/A')}
    - Recommended Treatments: {treatment_str}
    
    ROLE 2: AGRICULTURAL ECONOMIST (Market Analyst)
    - You are an expert on {crop_name} market trends, pricing strategies, and supply chains.
    - You know about the global and local (Indian) demand for {crop_name}.
    
    ROLE 3: AGRONOMIST (Yield Estimator)
    - You can estimate the potential yield for {crop_name}.
    - **CRITICAL:** If the user asks for a yield prediction but hasn't given their Farm Size (acres/hectares) or Region, YOU MUST ASK THEM FOR IT first.
    - Once you have the details, provide a realistic estimate range (e.g., "For 1 acre of {crop_name} in this region, average yield is X tons").
    
    *** CRITICAL LANGUAGE INSTRUCTION ***
    The user has selected their language as: {target_language}.
    You MUST provide your entire answer STRICTLY in {target_language}.
    - Do not mix languages (e.g., do not write Hindi in English script).
    - If the user asks in English but their selected language is {target_language}, TRANSLATE your thought process and reply ONLY in {target_language}.
    - Use the native script for the language (Devanagari for Hindi/Marathi, Kannada script for Kannada).
    
    INSTRUCTIONS:
    1. If the user asks about the **disease** (symptoms, cure, prevention), answer as the **Doctor** using the specific context above.
    2. If the user asks about **profit, market prices, selling, or business**, answer as the **Economist**.
       - Suggest the best time of year to sell {crop_name}.
       - Suggest value-added products.
    3. If the user asks about **yield, harvest, or production**, answer as the **Agronomist**.
       - **Rule:** If Farm Size or Region is missing, ask for it.
       - Important: Since the plant is diagnosed with {disease_key.replace('_', ' ') if disease_key else 'Unknown'}, mention that yield might be reduced by 10-20% if left untreated.
    4. Keep answers concise, helpful, and professional.
    """
    # --- 3. CALL GROQ API ---
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            # Use the "Instant" model for max speed
            model="llama-3.1-8b-instant",
            
            # Optional parameters to control creativity vs precision
            temperature=0.6,
            max_tokens=500,
        )
        
        # Extract the answer
        bot_reply = chat_completion.choices[0].message.content
        return jsonify({"response": bot_reply})
        
    except Exception as e:
        print(f"⚠️ Groq API Error: {e}")
        return jsonify({"error": "Dr. Crop is currently unavailable. Please try again later."}), 500
    
@app.route('/check_weather_risk', methods=['POST'])
def check_weather_risk():
    city = request.json.get('city')
    
    if not city:
        return jsonify({'error': "Please enter a city name."})

    # 1. Call OpenWeatherMap API
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            return jsonify({'error': "City not found!"})

        # 2. Extract Data
        temp = data['main']['temp']       # Temperature in Celsius
        humidity = data['main']['humidity'] # Humidity %
        condition = data['weather'][0]['description'] # e.g., "light rain"

        # 3. Analyze Disease Risk (Mapped to ALL 38 Classes)
        risk_level = "Low"
        alert_message = "Weather appears favorable for crop growth. Continue regular monitoring."
        
        # --- LOGIC 1: HIGH MOISTURE (Fungal & Bacterial Risks) ---
        # Triggers: High Humidity (>70%) OR Rain/Drizzle/Mist
        # Mapped Classes:
        # - Apple: Scab, Black Rot, Cedar Rust
        # - Cherry/Squash: Powdery Mildew
        # - Corn: Cercospora, Common Rust, Northern Blight
        # - Grape: Black Rot, Esca, Leaf Blight
        # - Peach/Pepper/Tomato: Bacterial Spots
        # - Potato/Tomato: Early & Late Blights, Leaf Mold, Septoria, Target Spot
        # - Strawberry: Leaf Scorch
        if humidity > 70 or any(x in condition.lower() for x in ['rain', 'drizzle', 'mist', 'thunderstorm']):
            risk_level = "HIGH (Fungal & Bacterial Risk)"
            alert_message = (
                f"⚠️ High Moisture ({humidity}% Humidity / {condition}) detected! "
                "This favors **Blights** (Potato/Tomato/Corn), **Rots** (Apple/Grape), "
                "**Rusts** (Corn/Apple), and **Bacterial Spots** (Pepper/Peach/Tomato). "
                "Action: Improve drainage and apply preventative fungicides if necessary."
            )
        
        # --- LOGIC 2: HOT & DRY (Pest Risks) ---
        # Triggers: High Temp (>30°C) AND Low Humidity (<50%)
        # Mapped Classes:
        # - Tomato: Spider Mites (Two-spotted spider mite) - They hate rain but love dry heat!
        elif temp > 30 and humidity < 50:
            risk_level = "Medium (Pest Risk)"
            alert_message = (
                f"⚠️ Hot & Dry conditions ({temp}°C, {humidity}% Humidity). "
                "This favors rapid reproduction of **Spider Mites** (especially on Tomatoes). "
                "Action: Check under leaves for webbing and mist foliage to increase humidity."
            )

        # --- LOGIC 3: WARM WEATHER (Viral Vector Risks) ---
        # Triggers: Warm Temps (>25°C) which make insects active
        # Mapped Classes:
        # - Orange: Huanglongbing (Spread by Psyllids)
        # - Tomato: Yellow Leaf Curl Virus, Mosaic Virus (Spread by Whiteflies/Aphids)
        elif temp > 25:
            risk_level = "Medium (Viral Vector Activity)"
            alert_message = (
                f"⚠️ Warm temperatures ({temp}°C) increase insect activity. "
                "Be noticeable of vectors spreading viruses like **Citrus Greening** (Orange) "
                "and **Yellow Leaf Curl** (Tomato). "
                "Action: Monitor for whiteflies and aphids."
            )

        # --- LOGIC 4: COLD STRESS ---
        # Triggers: Temp < 10°C
        # Mapped Classes: General health of warm crops (Tomato, Pepper, Corn, Squash)
        elif temp < 10:
             risk_level = "Medium (Cold Stress)"
             alert_message = "Temperatures are low. Frost damage is possible for sensitive crops like Tomato, Pepper, Corn, and Squash."

        return jsonify({
            'city': city,
            'temp': temp,
            'humidity': humidity,
            'condition': condition,
            'risk_level': risk_level,
            'alert_message': alert_message
        })

    except Exception as e:
        print(f"Weather API Error: {e}")
        return jsonify({'error': "Could not fetch weather data."})
    
# --- COMMUNITY & EXPERT ROUTES ---

@app.route('/community')
def community_page():
    with get_db_connection() as conn:
        posts = conn.execute("SELECT * FROM community_posts ORDER BY timestamp DESC").fetchall()
    return render_template('community.html', posts=posts, dark_mode=get_dark_mode())

@app.route('/ask_expert', methods=['POST'])
def ask_expert():
    data = request.json
    # We copy the existing image file to a safe 'community' reference
    # In this simple version, we reuse the same filename from uploads
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO community_posts (image_filename, crop_type, predicted_disease, confidence, user_question, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['filename'],
                data['crop_type'],
                data['disease_name'],
                data['confidence'],
                data.get('question', 'Is this diagnosis correct?'),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/expert_reply', methods=['POST'])
def expert_reply():
    post_id = request.form.get('post_id')
    reply_text = request.form.get('reply_text')
    reply_author = request.form.get('reply_author') or "Community Member"
    expert_code = request.form.get('expert_code')

    if expert_code and expert_code.strip():
        with get_db_connection() as conn:
            token_row = conn.execute("SELECT * FROM expert_tokens WHERE token = ?", (expert_code.strip(),)).fetchone()
        
        if token_row:
            # ✅ VALID TOKEN FOUND! 
            # Force the name to match the Verified Expert Name (e.g., "Dr. Ayush")
            reply_author = token_row['assigned_to_name']
        else:
            # ❌ INVALID TOKEN: Strip any fake titles user tried to add
            reply_author = reply_author.replace("Dr.", "").replace("Expert", "").strip()
    
    # 2. If NO token is provided, but they tried to use a title -> Block it
    elif "Dr." in reply_author or "Expert" in reply_author:
        reply_author = reply_author.replace("Dr.", "").replace("Expert", "").strip()

    # Final cleanup if name becomes empty
    if not reply_author: reply_author = "Community Member"
    
    # 3. Save the reply
    with get_db_connection() as conn:
        conn.execute("UPDATE community_posts SET expert_reply = ?, reply_author = ? WHERE id = ?", 
                     (reply_text, reply_author, post_id))
        conn.commit()
    
    return redirect(url_for('community_page'))

@app.route('/apply_for_expert', methods=['POST'])
def apply_for_expert():
    name = request.form.get('name')
    email = request.form.get('email')
    qualification = request.form.get('qualification')
    role = request.form.get('role') # Get the selected role
    
    with get_db_connection() as conn:
        conn.execute("INSERT INTO verification_requests (name, email, qualification, role, timestamp) VALUES (?, ?, ?, ?, ?)",
                     (name, email, qualification, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    
    return redirect(url_for('community_page'))

# --- UPDATED ADMIN DASHBOARD ---
@app.route('/admin_dashboard')
def admin_dashboard():
    with get_db_connection() as conn:
        requests = conn.execute("SELECT * FROM verification_requests WHERE status='pending'").fetchall()
        experts = conn.execute("SELECT * FROM expert_tokens").fetchall()
        
    html = """
    <html><body style='font-family:sans-serif; padding:40px; background:#f0f9ff;'>
        <h1>🛡️ Admin Verification Panel</h1>
        
        <h2>📝 Pending Applications</h2>
        <table border='1' cellspacing='0' cellpadding='10' style='background:white; width:100%; text-align:left;'>
            <tr style='background:#ddd;'><th>Name</th><th>Role</th><th>Email</th><th>ID/License</th><th>Action</th></tr>
            {rows}
        </table>
        
        <h2>✅ Active Experts</h2>
        <ul>{expert_list}</ul>
    </body></html>
    """
    
    rows_html = ""
    for r in requests:
        rows_html += f"""
        <tr>
            <td>{r['name']}</td>
            <td><b>{r['role']}</b></td> <td>{r['email']}</td>
            <td>{r['qualification']}</td>
            <td>
                <a href='/approve_expert/{r['id']}' style='background:green; color:white; padding:5px 10px; text-decoration:none; border-radius:5px;'>Approve</a>
            </td>
        </tr>
        """
        
    expert_list_html = "".join([f"<li><b>{e['assigned_to_name']}</b> - Token: <code>{e['token']}</code></li>" for e in experts])
    
    return html.format(rows=rows_html, expert_list=expert_list_html)

# --- UPDATED APPROVAL LOGIC ---
@app.route('/approve_expert/<int:req_id>')
def approve_expert(req_id):
    with get_db_connection() as conn:
        req = conn.execute("SELECT * FROM verification_requests WHERE id = ?", (req_id,)).fetchone()
        
        if req:
            # 1. SMART NAME GENERATION
            # If Doctor -> "Dr. Name"
            # If Shop Owner -> "Expert Name"
            if req['role'] == 'Doctor':
                final_name = "Dr. " + req['name']
            else:
                final_name = "Expert " + req['name'] + " (" + req['role'] + ")"

            # 2. Generate Token
            new_token = "EXP-" + secrets.token_hex(3).upper()
            
            # 3. Save to Database
            conn.execute("INSERT INTO expert_tokens (token, assigned_to_name) VALUES (?, ?)", (new_token, final_name))
            conn.execute("UPDATE verification_requests SET status='approved' WHERE id = ?", (req_id,))
            conn.commit()

            # 4. 👇 NEW: SEND EMAIL AUTOMATICALLY 👇
            # This calls the helper function we created earlier
            email_status = send_token_email(req['email'], final_name, new_token, req['role'])
            
            # Check if email worked or failed
            if email_status:
                status_msg = "✅ Email Sent Successfully!"
                status_color = "green"
            else:
                status_msg = "⚠️ Database updated, but Email failed (Check console logs)."
                status_color = "orange"

            # 5. Show Success Page
            return f"""
            <div style='text-align:center; margin-top:50px; font-family:sans-serif;'>
                <h1 style='color:green;'>✅ Verified Successfully!</h1>
                <p>Role: <b>{req['role']}</b></p>
                <p>Display Name: <b>{final_name}</b></p>
                <hr style='width:300px;'>
                <h3 style='color:{status_color};'>{status_msg}</h3>
                <p>Sent to: <b>{req['email']}</b></p>
                <br>
                <div style="background:#f3f4f6; padding:10px; display:inline-block; border-radius:10px;">
                    Token: <b>{new_token}</b>
                </div>
                <br><br>
                <a href='/admin_dashboard'>Return to Dashboard</a>
            </div>
            """
            
    return "Error: Request not found"

@app.route('/generate_expert_key/<name>')
def generate_expert_key(name):
    # This generates a random secure key like "EXP-a1b2c3d4"
    new_token = "EXP-" + secrets.token_hex(4)
    
    try:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO expert_tokens (token, assigned_to_name) VALUES (?, ?)", (new_token, name))
            conn.commit()
        return f"✅ Key Generated for {name}: <b>{new_token}</b><br>Give this key to the doctor."
    except Exception as e:
        return f"Error: {e}"

@app.route('/delete_community_post/<int:post_id>', methods=['POST'])
def delete_community_post(post_id):
    try:
        with get_db_connection() as conn:
            # We ONLY delete the database record. 
            # We do NOT delete the image file, as it might be used in History.
            conn.execute("DELETE FROM community_posts WHERE id = ?", (post_id,))
            conn.commit()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/schemes', methods=['GET', 'POST'])
def schemes_page():
    # Default filters
    selected_state = 'All'
    selected_crop = 'All'
    
    # If user filters via Form
    if request.method == 'POST':
        selected_state = request.form.get('state', 'All')
        selected_crop = request.form.get('crop', 'All')

    query = "SELECT * FROM government_schemes WHERE 1=1"
    params = []

    # Logic: If specific state chosen, show Global ('All') + That State
    # Logic: If specific state chosen, show Global ('All') + partial matches
    if selected_state != 'All':
        # CHANGED '=' TO 'LIKE' AND ADDED WILDCARDS (%)
        query += " AND (eligible_state = 'All' OR eligible_state LIKE ?)"
        params.append(f'%{selected_state}%')
        
    # Logic: Filter by crop
    if selected_crop != 'All':
        # CHANGED '=' TO 'LIKE' AND ADDED WILDCARDS (%)
        query += " AND (eligible_crop = 'All' OR eligible_crop LIKE ?)"
        params.append(f'%{selected_crop}%')
        
    with get_db_connection() as conn:
        schemes = conn.execute(query, params).fetchall()

    # Calculate days left for alerts
    today = datetime.now().date()
    final_schemes = []
    for s in schemes:
        scheme = dict(s)
        try:
            # Calculate days until deadline
            deadline_date = datetime.strptime(scheme['deadline'], '%Y-%m-%d').date()
            days_left = (deadline_date - today).days
        
            scheme['days_left'] = days_left
            scheme['is_urgent'] = 0 <= days_left <= 30
            scheme['is_expired'] = days_left < 0
        except ValueError:
            scheme['days_left'] = 0
            scheme['is_urgent'] = False
            scheme['is_expired'] = False
        
        
        
        final_schemes.append(scheme)

    return render_template('schemes.html', 
                           schemes=final_schemes, 
                           dark_mode=get_dark_mode(),
                           selected_state=selected_state,
                           selected_crop=selected_crop)


@app.route('/set_language/<language>')
def set_language(language):
    if language in app.config['BABEL_SUPPORTED_LOCALES']:
        session['language'] = language
    return redirect(request.referrer or '/')

@app.route('/api/update_sensor', methods=['POST'])
def receive_sensor_data():
    try:
        data = request.json
        # Extract data with defaults
        device_id = data.get('device_id', 'Unknown')
        moisture = data.get('moisture', 0)
        temp = data.get('temperature', 0)
        humid = data.get('humidity', 0)
        pump = data.get('pump_status', 'OFF') # New field

        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO soil_readings (device_id, moisture, temperature, humidity, pump_status)
                VALUES (?, ?, ?, ?, ?)
            ''', (device_id, moisture, temp, humid, pump))
            conn.commit()
            
        return jsonify({"status": "success", "message": "Data saved"}), 201
    except Exception as e:
        print(f"Sensor Error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
    
@app.route('/api/get_sensor_history')
def get_sensor_history():
    try:
        with get_db_connection() as conn:
            # Get the last 20 readings for the graph
            rows = conn.execute("SELECT * FROM soil_readings ORDER BY id DESC LIMIT 20").fetchall()
        
        # Reverse to show oldest -> newest (Left to Right on graph)
        data = [dict(row) for row in reversed(rows)]
        return jsonify(data)
    except Exception as e:
        return jsonify([])

@app.route('/sensors')
def sensors_page():
    return render_template('sensors.html', dark_mode=get_dark_mode())

@app.route('/api/disease_heatmap')
def get_heatmap_data():
    with get_db_connection() as conn:
        # Get data from last 30 days only (trends change!)
        rows = conn.execute('''
            SELECT disease_name, crop_type, latitude, longitude, city, timestamp 
            FROM disease_heatmap 
            WHERE date(timestamp) >= date('now', '-30 days')
        ''').fetchall()
    
    data = [dict(row) for row in rows]
    return jsonify(data)

@app.route('/api/update_location', methods=['POST'])
def update_location():
    try:
        lat = request.form.get('latitude')
        long = request.form.get('longitude')
        city = request.form.get('city')
        disease = request.form.get('disease')
        crop = request.form.get('crop')
        
        # Only save to Heatmap if it's a disease
        if lat and long and disease and "healthy" not in disease.lower():
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO disease_heatmap (disease_name, crop_type, latitude, longitude, city, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (disease, crop, float(lat), float(long), city, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
            return jsonify(success=True)
            
        return jsonify(success=False, message="Healthy crop or missing data")
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/heatmap')
def heatmap_page():
    return render_template('heatmap.html', dark_mode=get_dark_mode())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)