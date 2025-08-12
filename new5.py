FileName: "MultipleFiles/new2.py"
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # Assuming you'll use TensorFlow for your model
from gtts import gTTS
import os
import time
import base64  # For embedding audio in Streamlit
import re  # For more flexible command parsing

# --- Configuration ---
AUDIO_FILE_PATH = "voice_response.mp3"

# --- Translations for multilingual support ---
translations = {
    "en": {
        "appTitle": "Crop Disease Detector",
        "appSubtitle": "Upload a photo of your crop to detect diseases and get solutions",
        "uploadInstructions": "Drag & drop your crop image here or click to select",
        "uploadBtn": "Select Image",
        "analyzeBtn": "Analyze Image",
        "analysisTitle": "Analysis Results",
        "identifiedDisease": "Identified Disease",
        "confidence": "Confidence",
        "symptoms": "Symptoms",
        "chemicalTreatment": "Chemical Treatment",
        "organicTreatment": "Organic Treatment",
        "prevention": "Prevention",
        "footerText": "© 2023 Crop Disease Detector | Helping farmers protect their crops",
        "voiceAssistantTitle": "Voice Assistant (Text-based Simulation)",
        "typeCommand": "Type your command here (e.g., 'upload image', 'what are the symptoms of leaf rust?', 'change language to Spanish'):",
        "processing": "Processing your command...",
        "analysisComplete": "Analysis complete. Disease detected: {disease_name} with {confidence} confidence.",
        "selectImagePrompt": "Please select your crop image for analysis.",
        "displayingHeatmap": "Displaying disease heatmap analysis.",
        "showingSolutions": "Here are the recommended solutions for the detected disease.",
        "unrecognizedCommand": "I didn't understand that. Can you please rephrase or ask about symptoms, treatments, or prevention?",
        "languageChanged": "Language changed to {lang_name}.",
        "languageNotSupported": "Sorry, {lang_name} is not supported. Supported languages are English, Spanish, French, Hindi, Chinese, and Tamil.",
        "voiceCommandsList": "Available commands:\n- 'upload image'\n- 'show solutions'\n- 'change language to [language name]'\n- 'what are the symptoms [of disease]?'\n- 'tell me about chemical treatment [for disease]'\n- 'how to prevent [disease]'\n- 'hello', 'thank you', 'goodbye'",
        "invalidImageWarning": "Please upload only leaf or crop images for analysis."  # New warning message
    },
    "te": {  # Telugu Translation
        "appTitle": "పంట వ్యాధి డిటెక్టర్",
        "appSubtitle": "వ్యాధులను గుర్తించడానికి మరియు పరిష్కారాలను పొందడానికి మీ పంట ఫోటోను అప్‌లోడ్ చేయండి",
        "uploadInstructions": "మీ పంట చిత్రాన్ని ఇక్కడకు లాగండి లేదా ఎంచుకోవడానికి క్లిక్ చేయండి",
        "uploadBtn": "చిత్రాన్ని ఎంచుకోండి",
        "analyzeBtn": "చిత్రాన్ని విశ్లేషించండి",
        "analysisTitle": "విశ్లేషణ ఫలితాలు",
        "identifiedDisease": "గుర్తించబడిన వ్యాధి",
        "confidence": "విశ్వాసం",
        "symptoms": "లక్షణాలు",
        "chemicalTreatment": "రసాయన చికిత్స",
        "organicTreatment": "సేంద్రీయ చికిత్స",
        "prevention": "నివారణ",
        "footerText": "© 2023 పంట వ్యాధి డిటెక్టర్ | రైతులు తమ పంటలను రక్షించుకోవడానికి సహాయపడుతుంది",
        "voiceAssistantTitle": "వాయిస్ అసిస్టెంట్ (టెక్స్ట్ ఆధారిత సిమ్యులేషన్)",
        "typeCommand": "మీ ఆదేశాన్ని ఇక్కడ టైప్ చేయండి (ఉదా. 'చిత్రాన్ని అప్‌లోడ్ చేయండి', 'ఆకు తుప్పు లక్షణాలు ఏమిటి?', 'భాషను స్పానిష్‌కు మార్చండి'):",
        "processing": "మీ ఆదేశాన్ని ప్రాసెస్ చేస్తోంది...",
        "analysisComplete": "విశ్లేషణ పూర్తయింది. గుర్తించబడిన వ్యాధి: {disease_name} {confidence} విశ్వాసంతో.",
        "selectImagePrompt": "దయచేసి విశ్లేషణ కోసం మీ పంట చిత్రాన్ని ఎంచుకోండి.",
        "displayingHeatmap": "వ్యాధి హీట్‌మ్యాప్ విశ్లేషణను ప్రదర్శిస్తోంది.",
        "showingSolutions": "గుర్తించబడిన వ్యాధికి సిఫార్సు చేయబడిన పరిష్కారాలు ఇక్కడ ఉన్నాయి.",
        "unrecognizedCommand": "నాకు అది అర్థం కాలేదు. దయచేసి మళ్ళీ చెప్పగలరా లేదా లక్షణాలు, చికిత్సలు లేదా నివారణ గురించి అడగగలరా?",
        "languageChanged": "భాష {lang_name} కు మార్చబడింది.",
        "languageNotSupported": "క్షమించండి, {lang_name} మద్దతు లేదు. మద్దతు ఉన్న భాషలు ఇంగ్లీష్, స్పానిష్, ఫ్రెంచ్, హిందీ, చైనీస్ మరియు తమిళం.",
        "voiceCommandsList": "అందుబాటులో ఉన్న ఆదేశాలు:\n- 'చిత్రాన్ని అప్‌లోడ్ చేయండి'\n- 'పరిష్కారాలను చూపించు'\n- 'భాషను [భాష పేరు] కు మార్చండి'\n- '[వ్యాధి] లక్షణాలు ఏమిటి?'\n- '[వ్యాధి] రసాయన చికిత్స గురించి చెప్పండి'\n- '[వ్యాధి] ఎలా నివారించాలి?'\n- 'హలో', 'ధన్యవాదాలు', 'వీడ్కోలు'",
        "invalidImageWarning": "దయచేసి విశ్లేషణ కోసం ఆకు లేదా పంట చిత్రాలను మాత్రమే అప్‌లోడ్ చేయండి."
    },
    "hi": {
        "appTitle": "फसल रोग डिटेक्टर",
        "appSubtitle": "रोगों का पता लगाने और समाधान प्राप्त करने के लिए अपनी फसल की एक तस्वीर अपलोड करें",
        "uploadInstructions": "अपनी फसल छवि को यहां खींचें और छोड़ें या चयन करने के लिए क्लिक करें",
        "uploadBtn": "छवि चुनें",
        "analyzeBtn": "छवि का विश्लेषण करें",
        "analysisTitle": "विश्लेषण परिणाम",
        "identifiedDisease": "पहचानी गई बीमारी",
        "confidence": "विश्वास",
        "symptoms": "लक्षण",
        "chemicalTreatment": "रासायनिक उपचार",
        "organicTreatment": "जैविक उपचार",
        "prevention": "निवारण",
        "footerText": "© 2023 फसल रोग डिटेक्टर | किसानों को अपनी फसलों की रक्षा करने में मदद करना",
        "voiceAssistantTitle": "वॉयस असिस्टेंट (टेक्स्ट-आधारित सिमुलेशन)",
        "typeCommand": "अपना कमांड यहां टाइप करें (उदा. 'छवि अपलोड करें', 'पत्ती के जंग के लक्षण क्या हैं?', 'भाषा बदलें स्पेनिश में'):",
        "processing": "आपके कमांड को संसाधित किया जा रहा है...",
        "analysisComplete": "विश्लेषण पूरा हुआ। बीमारी का पता चला: {disease_name} {confidence} विश्वास के साथ।",
        "selectImagePrompt": "कृपया विश्लेषण के लिए अपनी फसल की छवि का चयन करें।",
        "displayingHeatmap": "रोग हीटमैप विश्लेषण प्रदर्शित किया जा रहा है।",
        "showingSolutions": "यहां पता चले रोग के लिए अनुशंसित समाधान दिए गए हैं।",
        "unrecognizedCommand": "मुझे वह समझ नहीं आया। क्या आप इसे फिर से कह सकते हैं या लक्षणों, उपचारों या रोकथाम के बारे में पूछ सकते हैं?",
        "languageChanged": "भाषा {lang_name} में बदल दी गई।",
        "languageNotSupported": "क्षमा करें, {lang_name} समर्थित नहीं है। समर्थित भाषाएँ अंग्रेजी, स्पेनिश, फ्रेंच, हिंदी, चीनी और तमिल हैं।",
        "voiceCommandsList": "उपलब्ध कमांड:\n- 'छवि अपलोड करें'\n- 'समाधान दिखाएं'\n- 'भाषा बदलें [भाषा का नाम]'\n- 'लक्षण क्या हैं [रोग के]?'\n- 'रासायनिक उपचार के बारे में बताएं [रोग के लिए]'\n- 'कैसे रोकें [रोग]'\n- 'नमस्ते', 'धन्यवाद', 'अलविदा'",
        "invalidImageWarning": "कृपया विश्लेषण के लिए केवल पत्ती या फसल की छवियां अपलोड करें."
    },
    "ta": {
        "appTitle": "பயிர் நோய் கண்டறிதல்",
        "appSubtitle": "நோய்களைக் கண்டறிந்து தீர்வுகளைப் பெற உங்கள் பயிரின் புகைப்படத்தைப் பதிவேற்றவும்",
        "uploadInstructions": "உங்கள் பயிர் படத்தை இங்கே இழுத்து விடவும் அல்லது தேர்ந்தெடுக்க கிளிக் செய்யவும்",
        "uploadBtn": "படத்தைத் தேர்ந்தெடுக்கவும்",
        "analyzeBtn": "படத்தை பகுப்பாய்வு செய்யவும்",
        "analysisTitle": "பகுப்பாய்வு முடிவுகள்",
        "identifiedDisease": "கண்டறியப்பட்ட நோய்",
        "confidence": "நம்பிக்கை",
        "symptoms": "அறிகுறிகள்",
        "chemicalTreatment": "இரசாயன சிகிச்சை",
        "organicTreatment": "இயற்கை சிகிச்சை",
        "prevention": "தடுப்பு",
        "footerText": "© 2023 பயிர் நோய் கண்டறிதல் | விவசாயிகள் தங்கள் பயிர்களைப் பாதுகாக்க உதவுகிறது",
        "voiceAssistantTitle": "குரல் உதவியாளர் (உரை அடிப்படையிலான உருவகப்படுத்துதல்)",
        "typeCommand": "உங்கள் கட்டளையை இங்கே தட்டச்சு செய்யவும் (எ.கா., 'படத்தைப் பதிவேற்றவும்', 'இலை துரு நோயின் அறிகுறிகள் என்ன?', 'மொழியை ஸ்பானிஷ் ஆக மாற்றவும்'):",
        "processing": "உங்கள் கட்டளையைச் செயலாக்குகிறது...",
        "analysisComplete": "பகுப்பாய்வு முடிந்தது. கண்டறியப்பட்ட நோய்: {disease_name} {confidence} நம்பிக்கையுடன்.",
        "selectImagePrompt": "பகுப்பாய்வுக்காக உங்கள் பயிர் படத்தை தேர்ந்தெடுக்கவும்.",
        "displayingHeatmap": "நோய் வெப்ப வரைபட பகுப்பாய்வைக் காட்டுகிறது.",
        "showingSolutions": "கண்டறியப்பட்ட நோய்க்கான பரிந்துரைக்கப்பட்ட தீர்வுகள் இங்கே.",
        "unrecognizedCommand": "எனக்கு அது புரியவில்லை. நீங்கள் மீண்டும் கூற முடியுமா அல்லது அறிகுறிகள், சிகிச்சைகள் அல்லது தடுப்பு பற்றி கேட்க முடியுமா?",
        "languageChanged": "மொழி {lang_name} ஆக மாற்றப்பட்டது.",
        "languageNotSupported": "மன்னிக்கவும், {lang_name} ஆதரிக்கப்படவில்லை. ஆதரிக்கப்படும் மொழிகள் ஆங்கிலம், ஸ்பானிஷ், பிரஞ்சு, இந்தி, சீனம் மற்றும் தமிழ்.",
        "voiceCommandsList": "கிடைக்கும் கட்டளைகள்:\n- 'படத்தைப் பதிவேற்றவும்'\n- 'தீர்வுகளைக் காட்டு'\n- 'மொழியை [மொழி பெயர்] ஆக மாற்றவும்'\n- '[நோய்] அறிகுறிகள் என்ன?'\n- '[நோய்] இரசாயன சிகிச்சை பற்றி சொல்லுங்கள்'\n- '[நோய்] தடுப்பது எப்படி?'\n- 'வணக்கம்', 'நன்றி', 'விடைபெறுகிறேன்'",
        "invalidImageWarning": "பகுப்பாய்வுக்காக இலை அல்லது பயிர் படங்களை மட்டுமே பதிவேற்றவும்."
    }
}

# --- Session State Initialization ---
if 'current_language' not in st.session_state:
    st.session_state.current_language = 'en'
if 'disease_info' not in st.session_state:
    st.session_state.disease_info = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'disease_model' not in st.session_state:
    st.session_state.disease_model = None
if 'crop_leaf_detector_model' not in st.session_state:
    st.session_state.crop_leaf_detector_model = None

# --- Helper Functions ---

# Function to speak text using gTTS and embed in Streamlit
def speak_text(text, lang):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(AUDIO_FILE_PATH)
        with open(AUDIO_FILE_PATH, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_html = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.components.v1.html(audio_html, height=0, width=0)
        # Give a small delay to ensure audio playback starts before file deletion
        time.sleep(0.5)
        if os.path.exists(AUDIO_FILE_PATH):
            os.remove(AUDIO_FILE_PATH)  # Clean up the audio file
    except Exception as e:
        st.error(f"Error playing audio: {e}. Please ensure your browser supports audio playback and try again.")

# Function to get translated text
def get_text(key):
    return translations[st.session_state.current_language].get(key, translations["en"][key])

# --- Model Loading (Placeholder) ---
@st.cache_resource  # Cache the model to avoid reloading on every rerun
def load_disease_model():
    """
    Loads the pre-trained crop disease detection model.
    Replace 'path/to/your/disease_model.h5' with the actual path to your model file.
    """
    try:
        # Example: Load a Keras/TensorFlow model
        # model = tf.keras.models.load_model('path/to/your/disease_model.h5')
        # return model
        # For demonstration, returning a dummy object
        st.success("Dummy Disease Model Loaded (replace with actual model)")
        return "dummy_disease_model"
    except Exception as e:
        st.error(f"Error loading disease model: {e}. Make sure the model file exists and is accessible.")
        return None

@st.cache_resource  # Cache the model to avoid reloading on every rerun
def load_crop_leaf_detector_model():
    """
    Loads the pre-trained model to detect if an image is a crop/leaf.
    Replace 'path/to/your/crop_leaf_detector_model.h5' with the actual path.
    """
    try:
        # Example: Load a Keras/TensorFlow model
        # model = tf.keras.models.load_model('path/to/your/crop_leaf_detector_model.h5')
        # return model
        # For demonstration, returning a dummy object
        st.success("Dummy Crop/Leaf Detector Model Loaded (replace with actual model)")
        return "dummy_crop_leaf_detector_model"
    except Exception as e:
        st.error(f"Error loading crop/leaf detector model: {e}. Make sure the model file exists and is accessible.")
        return None

# Initialize models on startup
if st.session_state.disease_model is None:
    st.session_state.disease_model = load_disease_model()
if st.session_state.crop_leaf_detector_model is None:
    st.session_state.crop_leaf_detector_model = load_crop_leaf_detector_model()

# --- Dummy Disease Database (Replace with actual model output mapping) ---
# In a real scenario, your model would output a class index,
# and you'd map that index to this detailed information.
disease_database = {
    "Leaf Rust": {
        "symptoms": "Small, round to oblong yellow spots on leaves that develop into reddish-brown pustules. The pustules are most common on the upper leaf surface and leaf sheaths.",
        "chemical_solution": "Apply fungicides containing propiconazole or tebuconazole at the first sign of disease, repeating every 14 days if conditions remain favorable for disease development.",
        "organic_solution": "Apply neem oil spray every 7-10 days. Remove and destroy infected leaves. Increase airflow and reduce leaf wetness by watering only at the base of plants.",
        "prevention_tips": "Plant resistant varieties when available. Ensure proper spacing (at least 18-24 inches apart) to promote air circulation. Avoid overhead watering. Clean up plant debris at the end of the season. Rotate crops if possible."
    },
    "Early Blight": {
        "symptoms": "Dark, concentric rings (target-like spots) on older leaves, often surrounded by a yellow halo. Can also affect stems and fruits.",
        "chemical_solution": "Fungicides containing chlorothalonil or mancozeb are effective. Apply at first sign of disease and repeat as per label instructions.",
        "organic_solution": "Use copper-based fungicides. Practice crop rotation. Ensure good air circulation. Remove and destroy infected plant parts immediately.",
        "prevention_tips": "Plant resistant varieties. Avoid overhead irrigation. Ensure proper plant spacing. Mulch around plants to prevent soil splash. Rotate crops every 2-3 years."
    },
    "Powdery Mildew": {
        "symptoms": "White, powdery spots on the surface of leaves, stems, and sometimes flowers. These spots can spread and cover entire leaves.",
        "chemical_solution": "Apply fungicides like myclobutanil or propiconazole. Ensure thorough coverage of affected areas.",
        "organic_solution": "Spray with a mixture of baking soda (1 tsp per liter of water) and a few drops of liquid soap. Neem oil is also effective. Ensure good air circulation.",
        "prevention_tips": "Plant resistant varieties. Provide good air circulation. Avoid excessive nitrogen fertilization. Water plants at the base to keep leaves dry."
    },
    "Healthy": {
        "symptoms": "Leaves are vibrant green, firm, and show no signs of discoloration, spots, or wilting. Plant growth is vigorous and uniform.",
        "chemical_solution": "No chemical treatment needed. Continue good agricultural practices.",
        "organic_solution": "No organic treatment needed. Maintain healthy soil and plant conditions.",
        "prevention_tips": "Continue with optimal watering, fertilization, and pest management. Monitor regularly for any early signs of stress or disease."
    }
}

# --- Placeholder for actual prediction logic ---
def predict_disease(image):
    """
    Simulates disease prediction. In a real app, this would use st.session_state.disease_model.
    Returns a tuple: (disease_name, confidence)
    """
    if st.session_state.disease_model == "dummy_disease_model":
        # Simulate different outcomes for demonstration
        if np.random.rand() < 0.7: # 70% chance of a disease
            diseases = list(disease_database.keys())
            diseases.remove("Healthy") # Don't randomly pick healthy if it's a disease
            predicted_disease = np.random.choice(diseases)
            confidence = round(0.7 + np.random.rand() * 0.2, 2) # 70-90% confidence
        else:
            predicted_disease = "Healthy"
            confidence = round(0.8 + np.random.rand() * 0.15, 2) # 80-95% confidence
        return predicted_disease, confidence
    else:
        # Here you would preprocess the image and feed it to your actual model
        # For example:
        # img = image.resize((224, 224)) # Resize for model input
        # img_array = np.array(img) / 255.0 # Normalize
        # img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        # predictions = st.session_state.disease_model.predict(img_array)
        # predicted_class_index = np.argmax(predictions)
        # confidence = np.max(predictions)
        # disease_name = your_class_labels[predicted_class_index]
        # return disease_name, confidence
        st.warning("Disease model not properly loaded. Using dummy prediction.")
        return "Unknown Disease", 0.5

def is_crop_leaf_image(image):
    """
    Simulates checking if the uploaded image is a crop or leaf.
    In a real app, this would use st.session_state.crop_leaf_detector_model.
    """
    if st.session_state.crop_leaf_detector_model == "dummy_crop_leaf_detector_model":
        # Simulate a check: assume it's a crop/leaf for now
        return True
    else:
        # Here you would preprocess the image and feed it to your actual detector model
        # For example:
        # img = image.resize((128, 128))
        # img_array = np.array(img) / 255.0
        # img_array = np.expand_dims(img_array, axis=0)
        # prediction = st.session_state.crop_leaf_detector_model.predict(img_array)
        # return prediction[0][0] > 0.5 # Assuming binary classification (crop/leaf vs. not)
        st.warning("Crop/Leaf detector model not properly loaded. Assuming image is valid.")
        return True

# --- Voice Assistant Command Processing ---
def process_command(command):
    command = command.lower().strip()
    response = ""
    lang = st.session_state.current_language

    if "upload image" in command:
        response = get_text("selectImagePrompt")
        # In a real app, you might trigger a file uploader or guide the user
        st.session_state.chat_history.append(f"**You:** {command}")
        st.session_state.chat_history.append(f"**Assistant:** {response}")
        speak_text(response, lang)
        return

    if "show solutions" in command:
        if st.session_state.disease_info:
            disease_name = st.session_state.disease_info["name"]
            response = get_text("showingSolutions").format(disease_name=disease_name)
            st.session_state.chat_history.append(f"**You:** {command}")
            st.session_state.chat_history.append(f"**Assistant:** {response}")
            speak_text(response, lang)
            # Display solutions in the main UI
            st.session_state.show_solutions = True
        else:
            response = get_text("selectImagePrompt") # Or "No disease analyzed yet."
            st.session_state.chat_history.append(f"**You:** {command}")
            st.session_state.chat_history.append(f"**Assistant:** {response}")
            speak_text(response, lang)
        return

    if "change language to" in command:
        match = re.search(r"change language to (\w+)", command)
        if match:
            lang_name = match.group(1).lower()
            supported_langs = {
                "english": "en", "spanish": "es", "french": "fr",
                "hindi": "hi", "chinese": "zh-cn", "tamil": "ta",
                "telugu": "te" # Added Telugu
            }
            if lang_name in supported_langs:
                st.session_state.current_language = supported_langs[lang_name]
                response = get_text("languageChanged").format(lang_name=lang_name.capitalize())
                st.session_state.chat_history.append(f"**You:** {command}")
                st.session_state.chat_history.append(f"**Assistant:** {response}")
                speak_text(response, st.session_state.current_language)
                st.experimental_rerun() # Rerun to update UI with new language
            else:
                response = get_text("languageNotSupported").format(lang_name=lang_name.capitalize())
                st.session_state.chat_history.append(f"**You:** {command}")
                st.session_state.chat_history.append(f"**Assistant:** {response}")
                speak_text(response, lang)
        else:
            response = get_text("unrecognizedCommand")
            st.session_state.chat_history.append(f"**You:** {command}")
            st.session_state.chat_history.append(f"**Assistant:** {response}")
            speak_text(response, lang)
        return

    # Commands related to disease info (symptoms, chemical, organic, prevention)
    disease_match = re.search(r"(symptoms|chemical treatment|organic treatment|prevent) (?:of|for)?\s*(.*)", command)
    if disease_match:
        info_type = disease_match.group(1)
        disease_query = disease_match.group(2).strip()

        # Try to find the disease in the database, prioritizing exact match or current detected disease
        target_disease = None
        if disease_query:
            for d_name in disease_database:
                if disease_query.lower() in d_name.lower():
                    target_disease = d_name
                    break
        elif st.session_state.disease_info:
            target_disease = st.session_state.disease_info["name"]

        if target_disease and target_disease in disease_database:
            info = disease_database[target_disease]
            if info_type == "symptoms":
                response = f"{target_disease} {get_text('symptoms').lower()}: {info['symptoms']}"
            elif info_type == "chemical treatment":
                response = f"{target_disease} {get_text('chemicalTreatment').lower()}: {info['chemical_solution']}"
            elif info_type == "organic treatment":
                response = f"{target_disease} {get_text('organicTreatment').lower()}: {info['organic_solution']}"
            elif info_type == "prevent":
                response = f"{target_disease} {get_text('prevention').lower()}: {info['prevention_tips']}"
            else:
                response = get_text("unrecognizedCommand")
        else:
            response = f"I don't have information for '{disease_query}' or no disease has been detected yet. Please specify a known disease or analyze an image first."
        st.session_state.chat_history.append(f"**You:** {command}")
        st.session_state.chat_history.append(f"**Assistant:** {response}")
        speak_text(response, lang)
        return

    if "hello" in command:
        response = "Hello! How can I assist you today?"
        st.session_state.chat_history.append(f"**You:** {command}")
        st.session_state.chat_history.append(f"**Assistant:** {response}")
        speak_text(response, lang)
        return

    if "thank you" in command or "thanks" in command:
        response = "You're welcome! Is there anything else I can help with?"
        st.session_state.chat_history.append(f"**You:** {command}")
        st.session_state.chat_history.append(f"**Assistant:** {response}")
        speak_text(response, lang)
        return

    if "goodbye" in command or "bye" in command:
        response = "Goodbye! Have a great day."
        st.session_state.chat_history.append(f"**You:** {command}")
        st.session_state.chat_history.append(f"**Assistant:** {response}")
        speak_text(response, lang)
        return

    response = get_text("unrecognizedCommand")
    st.session_state.chat_history.append(f"**You:** {command}")
    st.session_state.chat_history.append(f"**Assistant:** {response}")
    speak_text(response, lang)


# Define a callback function to handle the command and clear the input
# This function is placed here, after process_command and before the UI layout,
# which is a good logical place for helper functions.
def handle_command_and_clear():
    if st.session_state.command_input:
        process_command(st.session_state.command_input)
        # Clear the text input by setting its session state key to an empty string
        # This works because it's done within a callback, before the next rerun.
        st.session_state.command_input = ""


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title=get_text("appTitle"),
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #28a745;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.5em;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .stFileUploader > div > div > button {
        background-color: #28a745;
        color: white;
        border-radius: 0.5em;
        padding: 0.5em 1em;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 0.5em;
        padding: 0.5em 1em;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .analysis-section {
        border: 1px solid #e0e0e0;
        border-radius: 0.5em;
        padding: 1.5em;
        margin-top: 2em;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 0.5em;
        padding: 1em;
        height: 300px;
        overflow-y: auto;
        background-color: #f8f9fa;
    }
    .chat-message {
        margin-bottom: 0.5em;
    }
    .chat-message.user {
        text-align: right;
        color: #007bff;
    }
    .chat-message.assistant {
        text-align: left;
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(f"<h1 class='main-header'>{get_text('appTitle')}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subheader'>{get_text('appSubtitle')}</p>", unsafe_allow_html=True)

# Language selection in sidebar
st.sidebar.header("Language / భాష / भाषा")
lang_options = {
    "English": "en",
    "తెలుగు (Telugu)": "te",
    "हिंदी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta"
}
selected_lang_name = st.sidebar.selectbox(
    "Select Language",
    options=list(lang_options.keys()),
    index=list(lang_options.values()).index(st.session_state.current_language),
    format_func=lambda x: x # Display full name in selectbox
)
if lang_options[selected_lang_name] != st.session_state.current_language:
    st.session_state.current_language = lang_options[selected_lang_name]
    st.experimental_rerun() # Rerun to apply language change

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header(get_text("uploadInstructions"))
    uploaded_file = st.file_uploader(get_text("uploadBtn"), type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button(get_text("analyzeBtn")):
            if st.session_state.uploaded_image:
                # Check if it's a crop/leaf image first
                if is_crop_leaf_image(st.session_state.uploaded_image):
                    with st.spinner(get_text("processing")):
                        disease_name, confidence = predict_disease(st.session_state.uploaded_image)
                        st.session_state.disease_info = {
                            "name": disease_name,
                            "confidence": confidence,
                            "details": disease_database.get(disease_name, {})
                        }
                        st.success(get_text("analysisComplete").format(
                            disease_name=disease_name,
                            confidence=f"{confidence*100:.2f}%"
                        ))
                        speak_text(get_text("analysisComplete").format(
                            disease_name=disease_name,
                            confidence=f"{confidence*100:.0f} percent"
                        ), st.session_state.current_language)
                        st.session_state.show_solutions = True # Automatically show solutions after analysis
                else:
                    st.warning(get_text("invalidImageWarning"))
                    speak_text(get_text("invalidImageWarning"), st.session_state.current_language)
            else:
                st.warning(get_text("selectImagePrompt"))
                speak_text(get_text("selectImagePrompt"), st.session_state.current_language)

    if st.session_state.disease_info and st.session_state.show_solutions:
        st.markdown(f"<div class='analysis-section'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{get_text('analysisTitle')}</h3>", unsafe_allow_html=True)
        st.write(f"**{get_text('identifiedDisease')}:** {st.session_state.disease_info['name']}")
        st.write(f"**{get_text('confidence')}:** {st.session_state.disease_info['confidence']*100:.2f}%")

        disease_details = st.session_state.disease_info['details']
        if disease_details:
            st.subheader(get_text('symptoms'))
            st.write(disease_details.get('symptoms', 'N/A'))

            st.subheader(get_text('chemicalTreatment'))
            st.write(disease_details.get('chemical_solution', 'N/A'))

            st.subheader(get_text('organicTreatment'))
            st.write(disease_details.get('organic_solution', 'N/A'))

            st.subheader(get_text('prevention'))
            st.write(disease_details.get('prevention_tips', 'N/A'))
        else:
            st.info("No detailed information available for this disease in the database.")
        st.markdown(f"</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<h3>{get_text('voiceAssistantTitle')}</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-container'>", unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.chat_history):
        if "**You:**" in msg:
            st.markdown(f"<p class='chat-message user'>{msg}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='chat-message assistant'>{msg}</p>", unsafe_allow_html=True)
    st.markdown(f"</div>", unsafe_allow_html=True)

    # The key for st.text_input must be unique and consistent
    # This line reads the current value of the text input
    command_input_value = st.text_input(get_text("typeCommand"), key="command_input")

    # Attach the callback function to the button
    # The on_click callback will be executed when the button is pressed
    st.button("Send Command", on_click=handle_command_and_clear)

    st.markdown("---")
    st.markdown(f"**{get_text('voiceCommandsList')}**")


st.markdown(f"<p style='text-align: center; margin-top: 3em; color: #6c757d;'>{get_text('footerText')}</p>", unsafe_allow_html=True)
