import streamlit as st
import pandas as pd
import joblib

# Load trained models
rf_model = joblib.load("random_forest_pesticide_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
pesticide_encoder = joblib.load("pesticide_encoder.pkl")

# Load crop-disease-symptom dataset
try:
    crop_disease_symptom_df = pd.read_csv("expanded_crops_diseases_dataset.csv")  # Ensure this dataset exists
except FileNotFoundError:
    st.error("Error: 'expanded_crops_diseases_dataset.csv' file not found! Please check the file path.")
    st.stop()

# Language dictionary
languages = {
    "English": {
        "title": "Crop Disease & Pesticide Recommendation",
        "select_language": "Choose Language",
        "select_crop": "Select Crop",
        "select_disease": "Select Disease",
        "select_symptoms": "Select Symptoms",
        "predict": "Predict Pesticide",
        "recommended": "Recommended Pesticide: ",
        "error": "Error: Unknown input value detected!"
    },
    "Telugu": {
        "title": "పంట వ్యాధి మరియు పురుగుమందు సిఫార్సు",
        "select_language": "భాషను ఎంచుకోండి",
        "select_crop": "పంటను ఎంచుకోండి",
        "select_disease": "వ్యాధిని ఎంచుకోండి",
        "select_symptoms": "లక్షణాలను ఎంచుకోండి",
        "predict": "పురుగుమందును ఊహించు",
        "recommended": "సిఫార్సు చేసిన పురుగుమందు: ",
        "error": "పొరపాటు: తెలియని లెక్కింపు విలువ కనుగొనబడింది!"
    },
    "Tamil": {
        "title": "பயிர் நோய் மற்றும் பூச்சிக்கொல்லி பரிந்துரை",
        "select_language": "மொழி தேர்ந்தெடுக்கவும்",
        "select_crop": "பயிரை தேர்வு செய்யவும்",
        "select_disease": "நோயை தேர்வு செய்யவும்",
        "select_symptoms": "அறிகுறிகளை தேர்வு செய்யவும்",
        "predict": "பூச்சிக்கொல்லியை கணிக்கவும்",
        "recommended": "பரிந்துரைக்கப்பட்ட பூச்சிக்கொல்லி: ",
        "error": "பிழை: தெரியாத உள்ளீடு கண்டறியப்பட்டது!"
    },
    "Hindi": {
        "title": "फसल रोग और कीटनाशक सिफारिश",
        "select_language": "भाषा चुनें",
        "select_crop": "फसल चुनें",
        "select_disease": "रोग चुनें",
        "select_symptoms": "लक्षण चुनें",
        "predict": "कीटनाशक का अनुमान लगाएं",
        "recommended": "सिफारिश किया गया कीटनाशक: ",
        "error": "त्रुटि: अज्ञात इनपुट मान पाया गया!"
    }
}
crop_translations = {
    "Apple": {"Telugu": "ఆపిల్", "Tamil": "ஆப்பிள்", "Hindi": "सेब"},
    "Wheat": {"Telugu": "గోధుమ", "Tamil": "கோதுமை", "Hindi": "गेहूं"},
    "Rice (Paddy)": {"Telugu": "వరి (ధాన్యం)", "Tamil": "நெல்", "Hindi": "धान"},
    "Cotton": {"Telugu": "పత్తి", "Tamil": "பருத்தி", "Hindi": "कपास"},
    "Sugarcane": {"Telugu": "చెరుకు", "Tamil": "கரும்பு", "Hindi": "गन्ना"},
    "Tomato": {"Telugu": "టమాట", "Tamil": "தக்காளி", "Hindi": "टमाटर"},
    "Potato": {"Telugu": "బంగాళాదుంప", "Tamil": "உருளைக்கிழங்கு", "Hindi": "आलू"},
    "Areca Nut": {"Telugu": "పుగాకాయ", "Tamil": "பாக்கு", "Hindi": "सुपारी"},
    "Arecanut": {"Telugu": "పుగాకాయ", "Tamil": "பாக்கு", "Hindi": "सुपारी"},
    "Arhar (Tur) / Pigeon Pea": {"Telugu": "కంది", "Tamil": "துவரை", "Hindi": "अरहर (तूर)"},
    "Bajra (Pearl Millet)": {"Telugu": "సజ్జలు", "Tamil": "கம்பு", "Hindi": "बाजरा"},
    "Banana": {"Telugu": "అరటి", "Tamil": "வாழை", "Hindi": "केला"},
    "Barley": {"Telugu": "జవ", "Tamil": "பார்லி", "Hindi": "जौ"},
    "Capsicum": {"Telugu": "దొంగ మిరప", "Tamil": "குடை மிளகாய்", "Hindi": "शिमला मिर्च"},
    "Cardamom": {"Telugu": "ఏలకులు", "Tamil": "ஏலக்காய்", "Hindi": "इलायची"},
    "Castor Seed": {"Telugu": "ఆముదం గింజ", "Tamil": "ஆமணக்கு விதை", "Hindi": "अरंडी का बीज"},
    "Chana (Chickpeas)": {"Telugu": "సెనగలు", "Tamil": "கடலை", "Hindi": "चना"},
    "Chili": {"Telugu": "మిరపకాయ", "Tamil": "மிளகாய்", "Hindi": "मिर्च"},
    "Chilli": {"Telugu": "మిరపకాయ", "Tamil": "மிளகாய்", "Hindi": "मिर्च"},
    "Coconut": {"Telugu": "కొబ్బరి", "Tamil": "தேங்காய்", "Hindi": "नारियल"},
    "Coffee": {"Telugu": "కాఫీ", "Tamil": "காபி", "Hindi": "कॉफ़ी"},
    "Coriander": {"Telugu": "ధనియాలు", "Tamil": "கொத்தமல்லி", "Hindi": "धनिया"},
    "Garlic": {"Telugu": "వెల్లుల్లి", "Tamil": "பூண்டு", "Hindi": "लहसुन"},
    "Ginger": {"Telugu": "అల్లం", "Tamil": "இஞ்சி", "Hindi": "अदरक"},
    "Grapes": {"Telugu": "ద్రాక్ష", "Tamil": "திராட்சை", "Hindi": "अंगूर"},
    "Groundnut": {"Telugu": "వేరుశెనగ", "Tamil": "நிலக்கடலை", "Hindi": "मूंगफली"},
    "Groundnut (Peanut)": {"Telugu": "వేరుశెనగ", "Tamil": "நிலக்கடலை", "Hindi": "मूंगफली"},
    "Jowar (Sorghum)": {"Telugu": "జొన్న", "Tamil": "சோளம்", "Hindi": "ज्वार"},
    "Jute": {"Telugu": "జ్యూట్", "Tamil": "சணல்", "Hindi": "जूट"},
    "Maize": {"Telugu": "మొక్కజొన్న", "Tamil": "மக்காச்சோளம்", "Hindi": "मक्का"},
    "Maize (Corn)": {"Telugu": "మొక్కజొన్న", "Tamil": "மக்காச்சோளம்", "Hindi": "मक्का"},
    "Mango": {"Telugu": "మామిడి", "Tamil": "மாம்பழம்", "Hindi": "आम"},
    "Moong (Green Gram)": {"Telugu": "పెసర", "Tamil": "பச்சைப்பயறு", "Hindi": "मूंग"},
    "Mustard": {"Telugu": "ఆవాలు", "Tamil": "கடுகு", "Hindi": "सरसों"},
    "Onion": {"Telugu": "ఉల్లిపాయ", "Tamil": "வெங்காயம்", "Hindi": "प्याज"},
    "Rubber": {"Telugu": "రబ్బరు", "Tamil": "ரப்பர்", "Hindi": "रबर"},
    "Sorghum (Jowar)": {"Telugu": "జొన్న", "Tamil": "சோளம்", "Hindi": "ज्वार"},
    "Soybean": {"Telugu": "సోయాబీన్", "Tamil": "சோயா", "Hindi": "सोयाबीन"},
    "Sunflower": {"Telugu": "సూర్యముఖి", "Tamil": "சூரியகாந்தி", "Hindi": "सूरजमुखी"},
    "Tea": {"Telugu": "టీ", "Tamil": "தேயிலை", "Hindi": "चाय"},
    "Tobacco": {"Telugu": "పొగాకు", "Tamil": "புகையிலை", "Hindi": "तंबाकू"},
    "Pearl Millet (Bajra)": {"Telugu": "సజ్జలు", "Tamil": "கம்பு", "Hindi": "बाजरा"},
    "Pomegranate": {"Telugu": "దానిమ్మ", "Tamil": "மாதுளை", "Hindi": "अनार"},
    "Pulses (Lentil, Chickpea, Pigeon Pea)": {"Telugu": "పప్పులు (మినుములు, సెనగలు, కంది)", "Tamil": "பருப்பு வகைகள்", "Hindi": "दालें"},
    "Ragi (Finger Millet)": {"Telugu": "రాగి", "Tamil": "கேழ்வரகு", "Hindi": "रागी"},
    "Turmeric": {"Telugu": "పసుపు", "Tamil": "மஞ்சள்", "Hindi": "हल्दी"},
    "Urad (Black Gram)": {"Telugu": "మినుములు", "Tamil": "உளுந்து", "Hindi": "उड़द"}
}
def translate_crops(crop_list, language):
    """Translate crop names to selected language"""
    if language == "English":
        return crop_list  # No translation needed
    return [crop_translations.get(crop, {}).get(language, crop) for crop in crop_list]

def reverse_translate_crop(selected_crop, language):
    """Convert translated crop name back to English for model processing"""
    for eng_crop, translations in crop_translations.items():
        if translations.get(language) == selected_crop:
            return eng_crop  # Return English name
    return selected_crop  # Return as is if no match found

def main():
    # Select language
    selected_language = st.selectbox("🌍 " + languages["English"]["select_language"], list(languages.keys()))
    lang = languages[selected_language]

    st.title(lang["title"])

    # Select Crop (Dropdown in Telugu, Tamil, or Hindi)
    crop_options = sorted(crop_disease_symptom_df["Crop"].unique())  # Unique crop names in English
    translated_crops = translate_crops(crop_options, selected_language)
    selected_crop = st.selectbox(lang["select_crop"], translated_crops)
    crop = reverse_translate_crop(selected_crop, selected_language)

    # Filter diseases based on selected crop
    filtered_diseases = crop_disease_symptom_df[crop_disease_symptom_df["Crop"] == crop]["Disease"].unique()
    disease = st.selectbox(lang["select_disease"], filtered_diseases)

    # Filter symptoms based on selected disease
    filtered_symptoms = crop_disease_symptom_df[
        (crop_disease_symptom_df["Crop"] == crop) & (crop_disease_symptom_df["Disease"] == disease)
    ]["Symptoms"].unique()
    symptoms = st.selectbox(lang["select_symptoms"], filtered_symptoms)

    if st.button(lang["predict"]):
        try:
            # Encode user input
            crop_encoded = label_encoders['Crop'].transform([crop])[0]
            disease_encoded = label_encoders['Disease'].transform([disease])[0]
            symptoms_encoded = label_encoders['Symptoms'].transform([symptoms])[0]
        
            # Make prediction
            input_data = pd.DataFrame([[crop_encoded, disease_encoded, symptoms_encoded]], 
                                      columns=['Crop', 'Disease', 'Symptoms'])
            prediction = rf_model.predict(input_data)
            pesticide = pesticide_encoder.inverse_transform(prediction)[0]
        
            st.success(f"{lang['recommended']} {pesticide}")
        
        except Exception as e:
            st.error(f"⚠️ {lang['error']} \n\n {e}")

if __name__ == "__main__":
    main()
