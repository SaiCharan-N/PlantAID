import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ==============================
# Load model safely
# ==============================
# compile=False avoids potential shape/tuple errors
model = load_model('models/plantAID_final.keras', compile=False)

# ==============================
# Class labels (17 classes)
# ==============================
class_names = [
    'Corn___Corn___Common_Rust',
    'Corn___Corn___Gray_Leaf_Spot',
    'Corn___Corn___Healthy',
    'Corn___Corn___Northern_Leaf_Blight',
    'Potato___Potato___Early_Blight',
    'Potato___Potato___Healthy',
    'Potato___Potato___Late_Blight',
    'Rice___Rice___Brown_Spot',
    'Rice___Rice___Healthy',
    'Rice___Rice___Leaf_Blast',
    'Rice___Rice___Neck_Blast',
    'Wheat___Wheat___Brown_Rust',
    'Wheat___Wheat___Healthy',
    'Wheat___Wheat___Yellow_Rust',
    'sugarcane___Bacterial Blight',
    'sugarcane___Healthy',
    'sugarcane___Red Rot'
]

# ==============================
# Remedies dictionary
# ==============================
# You can fill details as needed; for now, healthy ones are simple
remedies = {
    # 🌽 CORN
    'Corn___Corn___Common_Rust': {
        "prevention": (
            "🛡️ Plant rust-resistant hybrids.\n"
            "🌾 Rotate corn crops yearly to prevent buildup of pathogens.\n"
            "🧹 Remove and destroy infected plant debris after harvest.\n"
            "💧 Ensure proper airflow and avoid overhead watering."
        ),
        "cure": (
            "💊 Apply fungicides like Mancozeb or Azoxystrobin early.\n"
            "🪴 Remove infected leaves promptly.\n"
            "💧 Maintain field hygiene to reduce spread."
        )
    },
    'Corn___Corn___Gray_Leaf_Spot': {
        "prevention": (
            "🌾 Rotate crops for 2–3 years to break disease cycle.\n"
            "🧹 Remove crop residue after harvest.\n"
            "💧 Avoid high humidity and excess nitrogen fertilization.\n"
            "🌿 Maintain proper spacing for airflow."
        ),
        "cure": (
            "🧴 Apply Strobilurin or Triazole fungicides early.\n"
            "🪴 Remove severely infected leaves.\n"
            "💧 Monitor and improve field drainage."
        )
    },
    'Corn___Corn___Northern_Leaf_Blight': {
        "prevention": (
            "🛡️ Use resistant corn varieties.\n"
            "🧹 Remove infected leaves and plant debris.\n"
            "🌾 Rotate with non-host crops.\n"
            "💧 Avoid late-season heavy watering."
        ),
        "cure": (
            "💧 Spray Propiconazole or Mancozeb when symptoms appear.\n"
            "🪴 Remove affected leaves promptly.\n"
            "🌿 Maintain weed-free fields."
        )
    },
    'Corn___Corn___Healthy': {
        "prevention": "🌱 Balanced nutrition, crop rotation, monitoring.",
        "cure": "✅ Healthy — continue proper irrigation and pest monitoring."
    },

    # 🥔 POTATO
    'Potato___Potato___Early_Blight': {
        "prevention": (
            "🌾 Rotate crops with non-solanaceous plants.\n"
            "💧 Avoid overwatering and excess nitrogen fertilization.\n"
            "🧹 Remove infected leaves promptly.\n"
            "🌿 Maintain proper spacing for airflow."
        ),
        "cure": (
            "💊 Apply Chlorothalonil or Mancozeb sprays early.\n"
            "🪴 Remove and destroy affected foliage.\n"
            "💧 Avoid overhead irrigation."
        )
    },
    'Potato___Potato___Late_Blight': {
        "prevention": (
            "🌧️ Avoid overhead irrigation and waterlogging.\n"
            "🛡️ Use resistant varieties.\n"
            "🌿 Maintain proper spacing for airflow.\n"
            "🧹 Remove and destroy infected plant parts."
        ),
        "cure": (
            "💧 Apply Ridomil Gold or Metalaxyl + Mancozeb.\n"
            "🪴 Remove affected foliage immediately.\n"
            "🌾 Monitor fields daily to prevent spread."
        )
    },
    'Potato___Potato___Black_Scurf': {
        "prevention": (
            "🪴 Use certified disease-free seed tubers.\n"
            "🌿 Ensure well-drained soil.\n"
            "💧 Avoid excessive moisture and reuse of contaminated soil.\n"
            "🌾 Rotate crops for at least 2 years."
        ),
        "cure": (
            "🧴 Treat seed tubers with fungicides before planting.\n"
            "🪴 Remove infected plants promptly.\n"
            "🌿 Maintain soil hygiene and crop rotation."
        )
    },
    'Potato___Potato___Healthy': {
        "prevention": "🌱 Use disease-free tubers and proper irrigation.",
        "cure": "✅ Healthy — maintain good soil drainage and sunlight exposure."
    },

    # 🌾 RICE
    'Rice___Rice___Brown_Spot': {
        "prevention": (
            "🌾 Use certified clean seeds.\n"
            "🪴 Ensure balanced nutrition, especially potassium and zinc.\n"
            "💧 Avoid drought stress and water stagnation.\n"
            "🌿 Maintain field sanitation."
        ),
        "cure": (
            "🧴 Apply Mancozeb or Tricyclazole 2–3 times at 10-day intervals.\n"
            "💧 Keep field moisture optimal.\n"
            "🪴 Remove infected leaves if possible."
        )
    },
    'Rice___Rice___Leaf_Blast': {
        "prevention": (
            "🔥 Avoid excess nitrogen fertilization.\n"
            "🌿 Use resistant varieties.\n"
            "🌾 Maintain proper spacing and rotate fields.\n"
            "💧 Keep leaf surfaces dry to prevent fungal growth."
        ),
        "cure": (
            "💧 Spray Tricyclazole or Isoprothiolane early.\n"
            "🪴 Remove infected leaves.\n"
            "🌾 Ensure proper drainage and airflow."
        )
    },
    'Rice___Rice___Neck_Blast': {
        "prevention": (
            "🌾 Plant resistant varieties.\n"
            "💧 Maintain proper water levels to reduce stress.\n"
            "🪴 Avoid dense planting.\n"
            "🧹 Remove and destroy infected straw."
        ),
        "cure": (
            "💊 Apply Tricyclazole or appropriate fungicides early.\n"
            "🧴 Protect healthy panicles.\n"
            "🌿 Monitor fields during flowering stage."
        )
    },
    'Rice___Rice___Healthy': {
        "prevention": "🌱 Irrigation, balanced fertilizer, weeding.",
        "cure": "✅ Healthy — continue proper field care."
    },

    # 🌾 WHEAT
    'Wheat___Wheat___Brown_Rust': {
        "prevention": (
            "🛡️ Use rust-resistant wheat varieties.\n"
            "🌾 Destroy volunteer wheat and crop residues.\n"
            "💧 Avoid excessive irrigation.\n"
            "🧹 Monitor fields regularly."
        ),
        "cure": (
            "💊 Apply Propiconazole or Mancozeb at early infection stage.\n"
            "🪴 Remove infected leaves promptly.\n"
            "🌿 Maintain field hygiene."
        )
    },
    'Wheat___Wheat___Yellow_Rust': {
        "prevention": (
            "🛡️ Plant early maturing resistant varieties.\n"
            "🌾 Rotate crops and avoid continuous wheat cropping.\n"
            "🧹 Remove infected plants promptly.\n"
            "💧 Avoid waterlogging and excessive nitrogen."
        ),
        "cure": (
            "💊 Apply Tebuconazole or Triazole fungicides early.\n"
            "🪴 Remove infected leaves.\n"
            "🌿 Monitor weekly and prevent spread."
        )
    },
    'Wheat___Wheat___Healthy': {
        "prevention": "🌱 Crop rotation and certified seeds.",
        "cure": "✅ Healthy — continue field monitoring."
    },

    # 🍬 SUGARCANE
    'sugarcane___Bacterial Blight': {
        "prevention": (
            "🧫 Use disease-free seeds and resistant varieties.\n"
            "💧 Avoid deep irrigation and high nitrogen fertilizer.\n"
            "🪴 Control leaf miners and wounds.\n"
            "🧹 Maintain sanitation in fields."
        ),
        "cure": (
            "💊 Apply Streptocycline (100–200 ppm) mixed with copper fungicides.\n"
            "💧 Drain water and allow field to dry.\n"
            "🪴 Remove and burn infected parts."
        )
    },
    'sugarcane___Red Rot': {
        "prevention": (
            "🔥 Use resistant varieties.\n"
            "💧 Avoid water stagnation.\n"
            "🧹 Disinfect cutting knives.\n"
            "🪴 Destroy affected clumps immediately."
        ),
        "cure": (
            "💊 Remove and burn infected stalks.\n"
            "💧 Drench soil with Bordeaux mixture around infected plants.\n"
            "🌿 Monitor fields regularly."
        )
    },
    'sugarcane___Smut': {
        "prevention": (
            "🌾 Use disease-free seed setts.\n"
            "💧 Treat with hot water (50°C for 30 min).\n"
            "🪴 Avoid ratooning infected fields.\n"
            "🧹 Maintain sanitation."
        ),
        "cure": (
            "🧴 Rogue out infected plants early.\n"
            "🌿 Treat soil with Trichoderma culture.\n"
            "💧 Maintain balanced nutrients."
        )
    },
    'sugarcane___Ratoon_Stunting': {
        "prevention": (
            "🪴 Use clean, hot-water-treated setts.\n"
            "🌾 Rotate with legume crops for at least one year.\n"
            "💧 Improve field drainage.\n"
            "🧹 Disinfect equipment."
        ),
        "cure": (
            "💧 Replace severely affected crops.\n"
            "🧴 Disinfect tools.\n"
            "🌿 Monitor remaining crops for signs of infection."
        )
    },
    'sugarcane___Healthy': {
        "prevention": "🌱 Balanced nutrition, sanitation, and crop rotation.",
        "cure": "✅ Healthy — continue pest-free monitoring."
    }
}


# ==============================
# In-memory history
# ==============================
history = []

# ==============================
# Prediction function
# ==============================
IMG_SIZE = 160  # match model training

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = round(float(np.max(preds)) * 100, 2)
    predicted_label = class_names[class_idx]
    remedy = remedies.get(predicted_label, {"prevention": "N/A", "cure": "N/A"})
    return predicted_label, confidence, remedy["prevention"], remedy["cure"]

# ==============================
# Flask routes
# ==============================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    if file:
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        label, confidence, prevention, cure = predict_disease(filepath)
        history.append({
            "filename": filename,
            "label": label,
            "confidence": confidence,
            "prevention": prevention,
            "cure": cure
        })
        return render_template(
            'result.html',
            filename=filename,
            label=label,
            confidence=confidence,
            prevention=prevention,
            cure=cure
        )

@app.route('/history')
def show_history():
    return render_template('history.html', records=history)

# ==============================
# Run Flask app
# ==============================
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
