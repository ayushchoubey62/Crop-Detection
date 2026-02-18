# 🌱 CropDoctor AI

**Smart Plant Disease Detection & Agronomy Assistant**

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Tech Stack](https://img.shields.io/badge/Stack-Flask%20|%20TensorFlow%20|%20Tailwind-blue)

**CropDoctor AI** is a full-stack agricultural platform designed to bridge the gap between advanced AI and rural farming. It features a **Hybrid AI Architecture** that allows disease detection to work both online (Cloud) and offline (Edge Device), ensuring farmers can protect their crops even in low-connectivity areas.

Beyond detection, it serves as a complete ecosystem with **Real-time Weather Risk Analysis**, a **GenAI Agronomist Chatbot**, and an **Expert Community Hub**.

---

## 🚀 Key Features

### 🧠 1. Hybrid AI Diagnosis Engine
* **Online Mode:** Utilizes a server-side **TensorFlow/Keras** model to detect 38+ classes of crop diseases with high accuracy.
* **Offline "Field Mode":** Seamless fallback to **TensorFlow.js** to run inference directly in the browser on the user's device (Edge AI). No internet required.

### 🛡️ 2. "Gatekeeper" Anti-Hallucination
* A secondary AI model validates uploaded images to ensure they are actually plants before processing, preventing false positives on irrelevant images.

### 🤖 3. "Dr. Crop" Context-Aware Chatbot
* Powered by **Groq (Llama-3)**.
* **Role-Playing AI:** Dynamically switches personas between **Pathologist** (Cure), **Economist** (Market Prices), and **Agronomist** (Yield Impact) based on the specific disease diagnosed.
* **Multilingual Voice Support:** Supports Speech-to-Text and Text-to-Speech in **English, Hindi, Marathi, and Kannada**.

### 🌦️ 4. Environmental Risk Analysis
* Integrates **OpenWeatherMap API** with custom heuristics.
* Analyzes live temperature and humidity to predict outbreaks (e.g., High Humidity → Fungal Risk alert).

### 🤝 5. Community & Expert Hub
* **Q&A Forum:** Farmers can post unresolved issues.
* **Expert Verification:** Admin dashboard to verify Agronomists and Doctors.
* **Green Badge System:** Verified experts receive special badges and capabilities.

### 📜 6. Resource Center
* **Government Schemes:** Searchable database of subsidies and insurance.
* **Disease Library:** Encyclopedia of symptoms and treatments.
* **PDF Reports:** Generate professional diagnosis reports instantly.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, Tailwind CSS, JavaScript, Jinja2.
* **Backend:** Python (Flask).
* **AI/ML:** TensorFlow (Keras), TensorFlow.js, Groq API (LLM).
* **Database:** SQLite.
* **External APIs:** OpenWeatherMap, Web Speech API.
* **Tools:** FPDF (Report Gen), Pillow (Image Processing).

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/cropdoctor-ai.git](https://github.com/yourusername/cropdoctor-ai.git)
cd cropdoctor-ai

2. Create Virtual Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
Bash
pip install -r requirements.txt

4. Set Up Environment Variables
Create a .env file in the root directory and add your API keys:
Code snippet
SECRET_KEY=your_random_secret_string
GROQ_API_KEY=your_groq_api_key_here
WEATHER_API_KEY=your_openweathermap_key_here
MAIL_USER=your_email@gmail.com
MAIL_PASS=your_app_password

5. Run the Application
Bash
python app.py

Access the app at http://127.0.0.1:5000.

🔮 Future Roadmap
[ ] Mobile App: Convert PWA to React Native for native mobile performance.

[ ] Cloud Storage: Migrate image storage to AWS S3.

[ ] WhatsApp Bot: Integrate Twilio for diagnosis via WhatsApp.

[ ] Drone Integration: API support for drone imagery analysis.

👨‍💻 Author
Ayush Choubey

LinkedIn: www.linkedin.com/in/ayush-choubey-b2a366281

GitHub: https://github.com/ayushchoubey62

Email: ayushchoubey800@gmail.com

Disclaimer: This tool is an AI assistant and should not replace professional agricultural advice.
