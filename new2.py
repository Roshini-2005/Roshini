import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import time
import base64 # For embedding audio in Streamlit
import re # For more flexible command parsing

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
        "voiceCommandsList": "Available commands:\n- 'upload image'\n- 'show solutions'\n- 'change language to [language name]'\n- 'what are the symptoms [of disease]?'\n- 'tell me about chemical treatment [for disease]'\n- 'how to prevent [disease]'\n- 'hello', 'thank you', 'goodbye'"
    },
    "es": {
        "appTitle": "Detector de Enfermedades de Cultivos",
        "appSubtitle": "Sube una foto de tu cultivo para detectar enfermedades y obtener soluciones",
        "uploadInstructions": "Arrastra y suelta tu imagen de cultivo aquí o haz clic para seleccionar",
        "uploadBtn": "Seleccionar Imagen",
        "analyzeBtn": "Analizar Imagen",
        "analysisTitle": "Resultados del Análisis",
        "identifiedDisease": "Enfermedad Identificada",
        "confidence": "Confianza",
        "symptoms": "Síntomas",
        "chemicalTreatment": "Tratamiento Químico",
        "organicTreatment": "Tratamiento Orgánico",
        "prevention": "Prevención",
        "footerText": "© 2023 Detector de Enfermedades de Cultivos | Ayudando a los agricultores a proteger sus cultivos",
        "voiceAssistantTitle": "Asistente de Voz (Simulación basada en texto)",
        "typeCommand": "Escribe tu comando aquí (ej. 'subir imagen', 'cuáles son los síntomas de la roya de la hoja?', 'cambiar idioma a inglés'):",
        "processing": "Procesando tu comando...",
        "analysisComplete": "Análisis completo. Enfermedad detectada: {disease_name} con {confidence} de confianza.",
        "selectImagePrompt": "Por favor, selecciona tu imagen de cultivo para el análisis.",
        "displayingHeatmap": "Mostrando el análisis del mapa de calor de la enfermedad.",
        "showingSolutions": "Aquí están las soluciones recomendadas para la enfermedad detectada.",
        "unrecognizedCommand": "No entendí eso. ¿Puedes reformular o preguntar sobre síntomas, tratamientos o prevención?",
        "languageChanged": "Idioma cambiado a {lang_name}.",
        "languageNotSupported": "Lo siento, {lang_name} no es compatible. Los idiomas compatibles son inglés, español, francés, hindi, chino y tamil.",
        "voiceCommandsList": "Comandos disponibles:\n- 'subir imagen'\n- 'mostrar soluciones'\n- 'cambiar idioma a [nombre del idioma]'\n- 'cuáles son los síntomas [de la enfermedad]?'\n- 'háblame del tratamiento químico [para la enfermedad]'\n- 'cómo prevenir [enfermedad]'\n- 'hola', 'gracias', 'adiós'"
    },
    "fr": {
        "appTitle": "Détecteur de Maladies des Cultures",
        "appSubtitle": "Téléchargez une photo de votre culture pour détecter les maladies et obtenir des solutions",
        "uploadInstructions": "Faites glisser et déposez votre image de culture ici ou cliquez para seleccionar",
        "uploadBtn": "Sélectionner une image",
        "analyzeBtn": "Analyser l'image",
        "analysisTitle": "Résultats d'analyse",
        "identifiedDisease": "Maladie identifiée",
        "confidence": "Confiance",
        "symptoms": "Symptômes",
        "chemicalTreatment": "Traitement chimique",
        "organicTreatment": "Traitement biologique",
        "prevention": "Prévention",
        "footerText": "© 2023 Détecteur de maladies des cultures | Aider les agriculteurs à protéger leurs cultures",
        "voiceAssistantTitle": "Assistant Vocal (Simulation textuelle)",
        "typeCommand": "Tapez votre commande ici (ex. 'télécharger l'image', 'quels sont les symptômes de la rouille des feuilles?', 'changer la langue en espagnol'):",
        "processing": "Traitement de votre commande...",
        "analysisComplete": "Analyse terminée. Maladie détectée : {disease_name} avec {confidence} de confiance.",
        "selectImagePrompt": "Veuillez sélectionner votre image de culture pour l'analyse.",
        "displayingHeatmap": "Affichage de l'analyse de la carte thermique des maladies.",
        "showingSolutions": "Voici les solutions recommandées pour la maladie détectée.",
        "unrecognizedCommand": "Je n'ai pas compris. Pouvez-vous reformuler ou poser des questions sur les symptômes, les traitements ou la prévention?",
        "languageChanged": "Langue changée en {lang_name}.",
        "languageNotSupported": "Désolé, {lang_name} n'est pas pris en charge. Les langues prises en charge sont l'anglais, l'espagnol, le français, l'hindi, le chinois et le tamil.",
        "voiceCommandsList": "Commandes disponibles :\n- 'télécharger l'image'\n- 'afficher les solutions'\n- 'changer la langue en [nom de la langue]'\n- 'quels sont les symptômes [de la maladie]?'\n- 'parlez-moi du traitement chimique [pour la maladie]'\n- 'comment prévenir [maladie]'\n- 'bonjour', 'merci', 'au revoir'"
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
        "voiceCommandsList": "उपलब्ध कमांड:\n- 'छवि अपलोड करें'\n- 'समाधान दिखाएं'\n- 'भाषा बदलें [भाषा का नाम]'\n- 'लक्षण क्या हैं [रोग के]?'\n- 'रासायनिक उपचार के बारे में बताएं [रोग के लिए]'\n- 'कैसे रोकें [रोग]'\n- 'नमस्ते', 'धन्यवाद', 'अलविदा'"
    },
    "zh": {
        "appTitle": "作物病害检测器",
        "appSubtitle": "上传您的作物照片以检测疾病并获得解决方案",
        "uploadInstructions": "将您的作物图片拖放到此处或单击以选择",
        "uploadBtn": "选择图片",
        "analyzeBtn": "分析图片",
        "analysisTitle": "分析结果",
        "identifiedDisease": "已识别的疾病",
        "confidence": "置信度",
        "symptoms": "症状",
        "chemicalTreatment": "化学处理",
        "organicTreatment": "有机处理",
        "prevention": "预防",
        "footerText": "© 2023 作物病害检测器 | 帮助农民保护他们的作物",
        "voiceAssistantTitle": "语音助手（基于文本的模拟）",
        "typeCommand": "在此处输入您的命令（例如：“上传图片”、“叶锈病的症状是什么？”、“将语言更改为西班牙语”）：",
        "processing": "正在处理您的命令...",
        "analysisComplete": "分析完成。检测到的疾病：{disease_name}，置信度为 {confidence}。",
        "selectImagePrompt": "请选择您的作物图片进行分析。",
        "displayingHeatmap": "正在显示疾病热图分析。",
        "showingSolutions": "以下是针对检测到的疾病的推荐解决方案。",
        "unrecognizedCommand": "我没听懂。请您重新措辞或询问症状、治疗或预防措施？",
        "languageChanged": "语言已更改为 {lang_name}。",
        "languageNotSupported": "抱歉，不支持 {lang_name}。支持的语言有英语、西班牙语、法语、印地语、中文和泰米尔语。",
        "voiceCommandsList": "可用命令：\n- '上传图片'\n- '显示解决方案'\n- '将语言更改为 [语言名称]'\n- '[疾病]的症状是什么？'\n- '告诉我[疾病]的化学治疗方法'\n- '如何预防[疾病]'\n- '你好', '谢谢', '再见'"
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
        "voiceCommandsList": "கிடைக்கும் கட்டளைகள்:\n- 'படத்தைப் பதிவேற்றவும்'\n- 'தீர்வுகளைக் காட்டு'\n- 'மொழியை [மொழி பெயர்] ஆக மாற்றவும்'\n- '[நோய்] அறிகுறிகள் என்ன?'\n- '[நோய்] இரசாயன சிகிச்சை பற்றி சொல்லுங்கள்'\n- '[நோய்] தடுப்பது எப்படி?'\n- 'வணக்கம்', 'நன்றி', 'விடைபெறுகிறேன்'"
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
        os.remove(AUDIO_FILE_PATH) # Clean up the audio file
    except Exception as e:
        st.error(f"Error playing audio: {e}. Please ensure your browser supports audio playback and try again.")

# Dummy function to simulate model prediction
def predict_disease(image):
    # In a real app, this would preprocess the image and call a TensorFlow model
    # For demonstration, we return a dummy response
    time.sleep(2) # Simulate processing time
    return {
        "name": "Leaf Rust",
        "confidence": "87%",
        "symptoms": "Small, round to oblong yellow spots on leaves that develop into reddish-brown pustules. The pustules are most common on the upper leaf surface and leaf sheaths.",
        "chemical_solution": "Apply fungicides containing propiconazole or tebuconazole at the first sign of disease, repeating every 14 days if conditions remain favorable for disease development.",
        "organic_solution": "Apply neem oil spray every 7-10 days. Remove and destroy infected leaves. Increase airflow and reduce leaf wetness by watering only at the base of plants.",
        "prevention_tips": "Plant resistant varieties when available. Ensure proper spacing (at least 18-24 inches apart) to promote air circulation. Avoid overhead watering. Clean up plant debris at the end of the season. Rotate crops if possible."
    }

# Function to process voice commands with more "AI-like" understanding
def process_voice_command(command):
    lang = st.session_state.current_language
    response_text = ""
    command_lower = command.lower().strip()
    disease_name_detected = st.session_state.disease_info['name'].lower() if st.session_state.disease_info else None

    # --- Language change commands ---
    lang_map = {
        "english": "en", "español": "es", "spanish": "es", "français": "fr", "french": "fr",
        "hindi": "hi", "हिन्दी": "hi", "chinese": "zh", "中文": "zh", "tamil": "ta", "தமிழ்": "ta"
    }
    for lang_key, lang_code in lang_map.items():
        if re.search(r"(change language to|cambiar idioma a|changer la langue en|भाषा बदलें|更改语言为|மொழியை மாற்றவும்)\s*" + re.escape(lang_key), command_lower):
            if lang_code in translations:
                st.session_state.current_language = lang_code
                response_text = translations[lang_code]["languageChanged"].format(lang_name=lang_key.capitalize())
                st.session_state.chat_history.append({"user": command, "bot": response_text})
                speak_text(response_text, lang_code)
                return
            else:
                response_text = translations[lang]["languageNotSupported"].format(lang_name=lang_key.capitalize())
                st.session_state.chat_history.append({"user": command, "bot": response_text})
                speak_text(response_text, lang)
                return

    # --- General conversational commands ---
    if re.search(r"^(hello|hi|hey|வணக்கம்)$", command_lower):
        response_text = "Hello! How can I assist you with your crop today?" if lang == "en" else \
                        "¡Hola! ¿Cómo puedo ayudarte con tu cultivo hoy?" if lang == "es" else \
                        "Bonjour! Comment puis-je vous aider avec votre culture aujourd'hui?" if lang == "fr" else \
                        "नमस्ते! मैं आज आपकी फसल में कैसे मदद कर सकता हूँ?" if lang == "hi" else \
                        "你好！今天我能如何帮助您的作物？" if lang == "zh" else \
                        "வணக்கம்! இன்று உங்கள் பயிருக்கு நான் எப்படி உதவ முடியும்?"
    elif re.search(r"^(thank you|thanks|நன்றி)$", command_lower):
        response_text = "You're welcome! Is there anything else I can help with?" if lang == "en" else \
                        "¡De nada! ¿Hay algo más en lo que pueda ayudarte?" if lang == "es" else \
                        "De rien! Y a-t-il autre chose que je puisse faire?" if lang == "fr" else \
                        "आपका स्वागत है! क्या मैं और कुछ मदद कर सकता हूँ?" if lang == "hi" else \
                        "不客气！还有什么可以帮您的吗？" if lang == "zh" else \
                        "வரவேற்கிறேன்! வேறு ஏதாவது நான் உதவ முடியுமா?"
    elif re.search(r"^(goodbye|bye|விடைபெறுகிறேன்)$", command_lower):
        response_text = "Goodbye! Have a great day protecting your crops." if lang == "en" else \
                        "¡Adiós! Que tengas un gran día protegiendo tus cultivos." if lang == "es" else \
                        "Au revoir! Passez une bonne journée à protéger vos cultures." if lang == "fr" else \
                        "अलविदा! अपनी फसलों की रक्षा करते हुए आपका दिन शानदार हो।" if lang == "hi" else \
                        "再见！祝您保护作物顺利。" if lang == "zh" else \
                        "விடைபெறுகிறேன்! உங்கள் பயிர்களைப் பாதுகாத்து ஒரு சிறந்த நாளைப் பெறுங்கள்."
    elif re.search(r"(upload image|subir imagen|télécharger l'image|छवि अपलोड करें|上传图片|படத்தைப் பதிவேற்றவும்)", command_lower):
        response_text = translations[lang]["selectImagePrompt"]
    elif re.search(r"(show solutions|mostrar soluciones|afficher les solutions|समाधान दिखाएं|显示解决方案|தீர்வுகளைக் காட்டு)", command_lower):
        if st.session_state.disease_info:
            response_text = translations[lang]["showingSolutions"]
        else:
            response_text = "Please upload an image and analyze it first to get solutions." if lang == "en" else \
                            "Por favor, sube una imagen y analízala primero para obtener soluciones." if lang == "es" else \
                            "Veuillez d'abord télécharger et analyser une image pour obtenir des solutions." if lang == "fr" else \
                            "समाधान प्राप्त करने के लिए कृपया पहले एक छवि अपलोड करें और उसका विश्लेषण करें।" if lang == "hi" else \
                            "请先上传图片并进行分析以获取解决方案。" if lang == "zh" else \
                            "தீர்வுகளைப் பெற முதலில் ஒரு படத்தைப் பதிவேற்றி பகுப்பாய்வு செய்யவும்."
    # --- Specific disease info commands ---
    elif re.search(r"(what are the symptoms|cuáles son los síntomas|quels sont les symptômes|लक्षण क्या हैं|症状是什么|அறிகுறிகள் என்ன)", command_lower):
        if st.session_state.disease_info:
            # Try to extract disease name from command, if not, use detected one
            match = re.search(r"(?:of|de|de la|du|के|的|இன்)\s*([\w\s]+)", command_lower)
            requested_disease = match.group(1).strip() if match else None

            if requested_disease and disease_name_detected and requested_disease in disease_name_detected:
                response_text = f"{translations[lang]['symptoms']}: {st.session_state.disease_info['symptoms']}"
            elif not requested_disease and disease_name_detected:
                 response_text = f"The symptoms for {st.session_state.disease_info['name']} are: {st.session_state.disease_info['symptoms']}" if lang == "en" else \
                                 f"Los síntomas de {st.session_state.disease_info['name']} son: {st.session_state.disease_info['symptoms']}" if lang == "es" else \
                                 f"Les symptômes de {st.session_state.disease_info['name']} sont : {st.session_state.disease_info['symptoms']}" if lang == "fr" else \
                                 f"{st.session_state.disease_info['name']} के लक्षण हैं: {st.session_state.disease_info['symptoms']}" if lang == "hi" else \
                                 f"{st.session_state.disease_info['name']}的症状是：{st.session_state.disease_info['symptoms']}" if lang == "zh" else \
                                 f"{st.session_state.disease_info['name']} க்கான அறிகுறிகள்: {st.session_state.disease_info['symptoms']}"
            else:
                response_text = "I need to analyze an image first to tell you about disease symptoms." if lang == "en" else \
                                "Necesito analizar una imagen primero para informarte sobre los síntomas de la enfermedad." if lang == "es" else \
                                "Je dois d'abord analyser une image pour vous parler des symptômes de la maladie." if lang == "fr" else \
                                "रोग के लक्षणों के बारे में बताने के लिए मुझे पहले एक छवि का विश्लेषण करना होगा।" if lang == "hi" else \
                                "我需要先分析图片才能告诉您疾病症状。" if lang == "zh" else \
                                "நோய் அறிகுறிகளைப் பற்றிச் சொல்ல முதலில் ஒரு படத்தைப் பகுப்பாய்வு செய்ய வேண்டும்."
        else:
            response_text = "I need to analyze an image first to tell you about disease symptoms." if lang == "en" else \
                            "Necesito analizar una imagen primero para informarte sobre los síntomas de la enfermedad." if lang == "es" else \
                            "Je dois d'abord analyser una imagen para vous parler des traitements chimiques." if lang == "fr" else \
                            "रोग के लक्षणों के बारे में बताने के लिए मुझे पहले एक छवि का विश्लेषण करना होगा।" if lang == "hi" else \
                            "我需要先分析图片才能告诉您疾病症状。" if lang == "zh" else \
                            "நோய் அறிகுறிகளைப் பற்றிச் சொல்ல முதலில் ஒரு படத்தைப் பகுப்பாய்வு செய்ய வேண்டும்."

    elif re.search(r"(tell me about chemical treatment|háblame del tratamiento químico|parlez-moi du traitement chimique|रासायनिक उपचार के बारे में बताएं|告诉我化学处理|இரசாயன சிகிச்சை பற்றி சொல்லுங்கள்)", command_lower):
        if st.session_state.disease_info:
            match = re.search(r"(?:for|para|pour|के लिए|的|கான)\s*([\w\s]+)", command_lower)
            requested_disease = match.group(1).strip() if match else None

            if requested_disease and disease_name_detected and requested_disease in disease_name_detected:
                response_text = f"{translations[lang]['chemicalTreatment']}: {st.session_state.disease_info['chemical_solution']}"
            elif not requested_disease and disease_name_detected:
                response_text = f"For {st.session_state.disease_info['name']}, the chemical treatment is: {st.session_state.disease_info['chemical_solution']}" if lang == "en" else \
                                f"Para {st.session_state.disease_info['name']}, el tratamiento químico es: {st.session_state.disease_info['chemical_solution']}" if lang == "es" else \
                                f"Pour {st.session_state.disease_info['name']}, le traitement chimique est : {st.session_state.disease_info['chemical_solution']}" if lang == "fr" else \
                                f"{st.session_state.disease_info['name']} के लिए, रासायनिक उपचार है: {st.session_state.disease_info['chemical_solution']}" if lang == "hi" else \
                                f"对于{st.session_state.disease_info['name']}，化学处理是：{st.session_state.disease_info['chemical_solution']}" if lang == "zh" else \
                                f"{st.session_state.disease_info['name']} க்கான இரசாயன சிகிச்சை: {st.session_state.disease_info['chemical_solution']}"
            else:
                response_text = "I need to analyze an image first to tell you about chemical treatments." if lang == "en" else \
                                "Necesito analizar una imagen primero para informarte sobre los tratamientos químicos." if lang == "es" else \
                                "Je dois d'abord analyser une image pour vous parler des traitements chimiques." if lang == "fr" else \
                                "रासायनिक उपचारों के बारे में बताने के लिए मुझे पहले एक छवि का विश्लेषण करना होगा।" if lang == "hi" else \
                                "我需要先分析图片才能告诉您化学治疗方法。" if lang == "zh" else \
                                "இரசாயன சிகிச்சைகள் பற்றிச் சொல்ல முதலில் ஒரு படத்தைப் பகுப்பாய்வு செய்ய வேண்டும்."
        else:
            response_text = "I need to analyze an image first to tell you about chemical treatments." if lang == "en" else \
                            "Necesito analizar una imagen primero para informarte sobre los tratamientos químicos." if lang == "es" else \
                            "Je dois d'abord analyser una imagen para vous parler des traitements chimiques." if lang == "fr" else \
                            "रासायनिक उपचारों के बारे में बताने के लिए मुझे पहले एक छवि का विश्लेषण करना होगा।" if lang == "hi" else \
                            "我需要先分析图片才能告诉您化学治疗方法。" if lang == "zh" else \
                            "இரசாயன சிகிச்சைகள் பற்றிச் சொல்ல முதலில் ஒரு படத்தைப் பகுப்பாய்வு செய்ய வேண்டும்."

    elif re.search(r"(how to prevent|cómo prevenir|comment prévenir|कैसे रोकें|如何预防|தடுப்பது எப்படி)", command_lower):
        if st.session_state.disease_info:
            match = re.search(r"(?:for|para|pour|के लिए|的|கான)\s*([\w\s]+)", command_lower)
            requested_disease = match.group(1).strip() if match else None

            if requested_disease and disease_name_detected and requested_disease in disease_name_detected:
                response_text = f"{translations[lang]['prevention']}: {st.session_state.disease_info['prevention_tips']}"
            elif not requested_disease and disease_name_detected:
                response_text = f"To prevent {st.session_state.disease_info['name']}: {st.session_state.disease_info['prevention_tips']}" if lang == "en" else \
                                f"Para prevenir {st.session_state.disease_info['name']}: {st.session_state.disease_info['prevention_tips']}" if lang == "es" else \
                                f"Pour prévenir {st.session_state.disease_info['name']} : {st.session_state.disease_info['prevention_tips']}" if lang == "fr" else \
                                f"{st.session_state.disease_info['name']} को रोकने के लिए: {st.session_state.disease_info['prevention_tips']}" if lang == "hi" else \
                                f"为了预防{st.session_state.disease_info['name']}：{st.session_state.disease_info['prevention_tips']}" if lang == "zh" else \
                                f"{st.session_state.disease_info['name']} ஐத் தடுக்க: {st.session_state.disease_info['prevention_tips']}"
            else:
                response_text = "I need to analyze an image first to tell you about prevention." if lang == "en" else \
                                "Necesito analizar una imagen primero para informarte sobre la prevención." if lang == "es" else \
                                "Je dois d'abord analyser une image pour vous parler de la prévention." if lang == "fr" else \
                                "रोकथाम के बारे में बताने के लिए मुझे पहले एक छवि का विश्लेषण करना होगा।" if lang == "hi" else \
                                "我需要先分析图片才能告诉您预防措施。" if lang == "zh" else \
                                "தடுப்பு பற்றிச் சொல்ல முதலில் ஒரு படத்தைப் பகுப்பாய்வு செய்ய வேண்டும்."
        else:
            response_text = "I need to analyze an image first to tell you about prevention." if lang == "en" else \
                            "Necesito analizar una imagen primero para informarte sobre la prevención." if lang == "es" else \
                            "Je debo analizar una imagen primero para informarte sobre la prevención." if lang == "fr" else \
                            "रोकथाम के बारे में बताने के लिए मुझे पहले एक छवि का विश्लेषण करना होगा।" if lang == "hi" else \
                            "我需要先分析图片才能告诉您预防措施。" if lang == "zh" else \
                            "தடுப்பு பற்றிச் சொல்ல முதலில் ஒரு படத்தைப் பகுப்பாய்வு செய்ய வேண்டும்."
    else:
        response_text = translations[lang]["unrecognizedCommand"]

    st.session_state.chat_history.append({"user": command, "bot": response_text})
    speak_text(response_text, lang)

# --- Streamlit UI ---

# Language Selector
st.sidebar.header("Language")
language_options = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "hi": "हिन्दी",
    "zh": "中文",
    "ta": "தமிழ்" # Added Tamil
}
selected_language_name = st.sidebar.selectbox(
    "Select Language",
    options=list(language_options.values()),
    index=list(language_options.keys()).index(st.session_state.current_language),
    key="lang_select_box"
)
# Update session state based on selected language name
for code, name in language_options.items():
    if name == selected_language_name:
        st.session_state.current_language = code
        break

lang = translations[st.session_state.current_language]

st.title(lang["appTitle"])
st.subheader(lang["appSubtitle"])

# --- Upload Section ---
st.header(lang["uploadInstructions"])
uploaded_file = st.file_uploader(lang["uploadBtn"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption='Uploaded Image', use_column_width=True)

    if st.button(lang["analyzeBtn"]):
        with st.spinner(lang["processing"]):
            st.session_state.disease_info = predict_disease(st.session_state.uploaded_image)
            response_text = lang["analysisComplete"].format(
                disease_name=st.session_state.disease_info['name'],
                confidence=st.session_state.disease_info['confidence']
            )
            st.session_state.chat_history.append({"user": lang["analyzeBtn"], "bot": response_text})
            speak_text(response_text, st.session_state.current_language)
        st.success("Analysis complete!")

# --- Analysis Section ---
if st.session_state.disease_info:
    st.header(lang["analysisTitle"])
    disease_info = st.session_state.disease_info

    st.markdown(f"**{lang['identifiedDisease']}:** <span style='color: red; font-weight: bold;'>{disease_info['name']}</span>", unsafe_allow_html=True)
    st.write(f"**{lang['confidence']}:** {disease_info['confidence']}")
    st.write(f"**{lang['symptoms']}:** {disease_info['symptoms']}")

    st.subheader(lang["chemicalTreatment"])
    st.write(disease_info['chemical_solution'])

    st.subheader(lang["organicTreatment"])
    st.write(disease_info['organic_solution'])

    st.subheader(lang["prevention"])
    st.write(disease_info['prevention_tips'])

# --- Voice Assistant Section (Chatbot) ---
st.sidebar.header(lang["voiceAssistantTitle"])

# Display chat history
for chat in st.session_state.chat_history:
    st.sidebar.markdown(f"**You:** {chat['user']}")
    st.sidebar.markdown(f"**AI:** {chat['bot']}")
    st.sidebar.markdown("---")

user_command = st.sidebar.text_input(lang["typeCommand"], key="voice_command_input")

if st.sidebar.button("Process Command"):
    if user_command:
        process_voice_command(user_command)
        # The chat history is updated within process_voice_command, and the UI will re-render
    else:
        response_text = "Please type a command."
        st.session_state.chat_history.append({"user": "", "bot": response_text}) # Add empty user command for clarity
        speak_text(response_text, st.session_state.current_language)

st.sidebar.markdown(f"**{lang['voiceCommandsList']}**")

# --- Footer ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{lang['footerText']}</p>", unsafe_allow_html=True)
