from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('Random_Forest.pkl')
mx = joblib.load('minmax_scaler.pkl')

# Define feature names - these should match exactly what was used during training
FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Define the crop dictionary
crop_dict = {
    'rice': 1,
    'maize': 2,
    'chickpea': 3,
    'kidneybeans': 4,
    'pigeonpeas': 5,
    'mothbeans': 6,
    'mungbean': 7,
    'blackgram': 8,
    'lentil': 9,
    'pomegranate': 10,
    'banana': 11,
    'mango': 12,
    'grapes': 13,
    'watermelon': 14,
    'muskmelon': 15,
    'apple': 16,
    'orange': 17,
    'papaya': 18,
    'coconut': 19,
    'cotton': 20,
    'jute': 21,
    'coffee': 22
}

# Create reverse mapping dictionary
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def calculate_wsi(N, P, K, temperature, humidity, ph, rainfall):
    # Your existing WSI calculation code
    min_values = {'N': 0, 'P': 0, 'K': 0, 'temperature': 0, 'humidity': 0, 'ph': 0, 'rainfall': 0}
    max_values = {'N': 100, 'P': 100, 'K': 100, 'temperature': 50, 'humidity': 100, 'ph': 14, 'rainfall': 300}
    
    norm_N = normalize(N, min_values['N'], max_values['N'])
    norm_P = normalize(P, min_values['P'], max_values['P'])
    norm_K = normalize(K, min_values['K'], max_values['K'])
    norm_temperature = normalize(temperature, min_values['temperature'], max_values['temperature'])
    norm_humidity = normalize(humidity, min_values['humidity'], max_values['humidity'])
    norm_ph = normalize(ph, min_values['ph'], max_values['ph'])
    norm_rainfall = normalize(rainfall, min_values['rainfall'], max_values['rainfall'])
    
    weights = {'N': 0.1, 'P': 0.1, 'K': 0.1, 'temperature': 0.2, 'humidity': 0.2, 'ph': 0.1, 'rainfall': 0.2}
    
    wsi = (weights['N'] * norm_N +
           weights['P'] * norm_P +
           weights['K'] * norm_K +
           weights['temperature'] * norm_temperature +
           weights['humidity'] * norm_humidity +
           weights['ph'] * norm_ph +
           weights['rainfall'] * norm_rainfall)
    
    return wsi

def generate_suggestions(N, P, K, temperature, humidity, ph, rainfall, wsi):
    suggestions = []
    tamil_suggestions = []
    hindi_suggestions = []
    hinglish_suggestions = []

    # Suggestions based on individual parameters
    if N <= 45:
        suggestions.append("Nitrogen levels are low. Consider using nitrogen-based fertilizers.")
        tamil_suggestions.append("நைட்ரஜன் அளவு குறைவாக உள்ளது. நைட்ரஜன் அடிப்படையிலான உரங்களைப் பயன்படுத்துவதைக் கவனியுங்கள்.")
        hindi_suggestions.append("नाइट्रोजन का स्तर कम है। नाइट्रोजन-आधारित उर्वरकों का उपयोग करने पर विचार करें।")
        hinglish_suggestions.append("Nitrogen ka star kam hai. Nitrogen-based fertilizers ka use karen.")
    elif N >= 105:
        suggestions.append("Nitrogen levels are high. Avoid additional nitrogen-based fertilizers.")
        tamil_suggestions.append("நைட்ரஜன் அளவு அதிகமாக உள்ளது. கூடுதல் நைட்ரஜன் அடிப்படையிலான உரங்களைத் தவிர்க்கவும்.")
        hindi_suggestions.append("नाइट्रोजन का स्तर अधिक है। अतिरिक्त नाइट्रोजन-आधारित उर्वरकों से बचें।")
        hinglish_suggestions.append("Nitrogen ka star zyada hai. Extra nitrogen-based fertilizers se bachen.")

    if P <= 45:
        suggestions.append("Phosphorus levels are low. Consider using phosphorus-based fertilizers.")
        tamil_suggestions.append("பாஸ்பரஸ் அளவு குறைவாக உள்ளது. பாஸ்பரஸ் அடிப்படையிலான உரங்களைப் பயன்படுத்துவதைக் கவனியுங்கள்.")
        hindi_suggestions.append("फास्फोरस का स्तर कम है। फास्फोरस-आधारित उर्वरकों का उपयोग करने पर विचार करें।")
        hinglish_suggestions.append("Phosphorus ka star kam hai. Phosphorus-based fertilizers ka use karen.")
    elif P >= 105:
        suggestions.append("Phosphorus levels are high. Avoid additional phosphorus-based fertilizers.")
        tamil_suggestions.append("பாஸ்பரஸ் அளவு அதிகமாக உள்ளது. கூடுதல் பாஸ்பரஸ் அடிப்படையிலான உரங்களைத் தவிர்க்கவும்.")
        hindi_suggestions.append("फास्फोरस का स्तर अधिक है। अतिरिक्त फास्फोरस-आधारित उर्वरकों से बचें।")
        hinglish_suggestions.append("Phosphorus ka star zyada hai. Extra phosphorus-based fertilizers se bachen.")

    if K <= 70:
        suggestions.append("Potassium levels are low. Consider using potassium-based fertilizers.")
        tamil_suggestions.append("பொட்டாசியம் அளவு குறைவாக உள்ளது. பொட்டாசியம் அடிப்படையிலான உரங்களைப் பயன்படுத்துவதைக் கவனியுங்கள்.")
        hindi_suggestions.append("पोटेशियम का स्तर कम है। पोटेशियम-आधारित उर्वरकों का उपयोग करने पर विचार करें।")
        hinglish_suggestions.append("Potassium ka star kam hai. Potassium-based fertilizers ka use karen.")
    elif K >= 160:
        suggestions.append("Potassium levels are high. Avoid additional potassium-based fertilizers.")
        tamil_suggestions.append("பொட்டாசியம் அளவு அதிகமாக உள்ளது. பொட்டாசியம் சார்ந்த கூடுதல் உரங்களைத் தவிர்க்கவும்.")
        hindi_suggestions.append("पोटेशियम का स्तर अधिक है। अतिरिक्त पोटेशियम-आधारित उर्वरकों से बचें।")
        hinglish_suggestions.append("Potassium ka star zyada hai. Extra potassium-based fertilizers se bachen.")

    if temperature <= 15:
        suggestions.append("Temperature is low. Consider using protective measures to maintain warmth.")
        tamil_suggestions.append("வெப்பநிலை குறைவாக உள்ளது. வெப்பத்தைத் தக்கவைக்க பாதுகாப்பு நடவடிக்கைகளைப் பயன்படுத்துவதைக் கவனியுங்கள்.")
        hindi_suggestions.append("तापमान कम है। गर्मी बनाए रखने के लिए सुरक्षात्मक उपायों का उपयोग करने पर विचार करें।")
        hinglish_suggestions.append("Temperature kam hai. Garmi banaye rakhne ke liye protective measures ka use karen.")
    elif temperature >= 35:
        suggestions.append("Temperature is high. Ensure adequate watering and consider shading.")
        tamil_suggestions.append("வெப்பநிலை அதிகமாக உள்ளது. போதுமான நீர்ப்பாசனத்தை உறுதிசெய்து, நிழலைக் கருத்தில் கொள்ளுங்கள்.")
        hindi_suggestions.append("तापमान अधिक है। पर्याप्त पानी दें और छायांकन पर विचार करें।")
        hinglish_suggestions.append("Temperature zyada hai. Pani den aur shading ka vichar karen.")

    if humidity <= 30:
        suggestions.append("Humidity is low. Ensure adequate irrigation to maintain soil moisture.")
        tamil_suggestions.append("ஈரப்பதம் குறைவாக உள்ளது. மண்ணின் ஈரப்பதத்தை பராமரிக்க போதுமான நீர்ப்பாசனத்தை உறுதி செய்யவும்.")
        hindi_suggestions.append("आर्द्रता कम है। मिट्टी की नमी बनाए रखने के लिए पर्याप्त सिंचाई सुनिश्चित करें।")
        hinglish_suggestions.append("Humidity kam hai. Mitti ki nami banaye rakhne ke liye irrigation ka use karen.")
    elif humidity >= 70:
        suggestions.append("Humidity is high. Ensure proper drainage to avoid waterlogging.")
        tamil_suggestions.append("ஈரப்பதம் அதிகமாக உள்ளது. நீர் தேங்குவதைத் தவிர்க்க சரியான வடிகால் வசதியை உறுதி செய்யவும்.")
        hindi_suggestions.append("आर्द्रता अधिक है। जलभराव से बचने के लिए उचित जल निकासी सुनिश्चित करें।")
        hinglish_suggestions.append("Humidity zyada hai. Jalbharav se bachne ke liye proper drainage ka use karen.")

    if ph < 6.5:
        suggestions.append("Soil pH is low (acidic). Consider using lime to raise the pH.")
        tamil_suggestions.append("மண்ணின் pH குறைவாக உள்ளது (அமிலத்தன்மை). pH ஐ உயர்த்த சுண்ணாம்பு பயன்படுத்துவதைக் கவனியுங்கள்.")
        hindi_suggestions.append("मिट्टी का पीएच कम है (अम्लीय)। पीएच बढ़ाने के लिए चूना का उपयोग करने पर विचार करें।")
        hinglish_suggestions.append("Mitti ka pH kam hai (amli). pH badhane ke liye lime ka use karen.")
    elif ph > 7.5:
        suggestions.append("Soil pH is high (alkaline). Consider using sulfur to lower the pH.")
        tamil_suggestions.append("மண்ணின் pH அதிகமாக உள்ளது (காரத்தன்மை). pH ஐக் குறைக்க கந்தகத்தைப் பயன்படுத்துவதைக் கவனியுங்கள்.")
        hindi_suggestions.append("मिट्टी का पीएच अधिक है (क्षारीय)। पीएच को कम करने के लिए सल्फर का उपयोग करने पर विचार करें।")
        hinglish_suggestions.append("Mitti ka pH zyada hai (kshari). pH kam karne ke liye sulfur ka use karen.")

    if rainfall <= 90:
        suggestions.append("Rainfall is low. Ensure adequate irrigation to supplement water needs.")
        tamil_suggestions.append("மழைப்பொழிவு குறைவாக உள்ளது. நீர் தேவைகளை பூர்த்தி செய்ய போதுமான பாசனத்தை உறுதி செய்யவும்.")
        hindi_suggestions.append("वर्षा कम है। पानी की आवश्यकताओं को पूरा करने के लिए पर्याप्त सिंचाई सुनिश्चित करें।")
        hinglish_suggestions.append("Rainfall kam hai. Pani ki zarurat ko pura karne ke liye irrigation ka use karen.")
    elif rainfall >= 210:
        suggestions.append("Rainfall is high. Ensure proper drainage to prevent waterlogging.")
        tamil_suggestions.append("மழைப்பொழிவு அதிகமாக உள்ளது. நீர் தேங்குவதைத் தடுக்க முறையான வடிகால் வசதியை உறுதி செய்யவும்.")
        hindi_suggestions.append("वर्षा अधिक है। जलभराव को रोकने के लिए उचित जल निकासी सुनिश्चित करें।")
        hinglish_suggestions.append("Rainfall zyada hai. Jalbharav rokne ke liye proper drainage ka use karen.")

    # General suggestions based on WSI
    if wsi < 0.3:
        suggestions.append("Weather conditions are favorable for crop growth.")
        tamil_suggestions.append("பருவநிலை பயிர் வளர்ச்சிக்கு சாதகமானது.")
        hindi_suggestions.append("मौसम की स्थिति फसल वृद्धि के लिए अनुकूल है।")
        hinglish_suggestions.append("Mausam ki sthiti fasal vridhi ke liye anukool hai.")
    elif 0.3 <= wsi <= 0.6:
        suggestions.append("Some weather stress detected. Monitor closely and adjust management practices.")
        tamil_suggestions.append("சில வானிலை அழுத்தம் கண்டறியப்பட்டது. மேலாண்மை நடைமுறைகளை உன்னிப்பாகக் கண்காணித்து சரிசெய்யவும்.")
        hindi_suggestions.append("कुछ मौसम तनाव का पता चला है। निकट से निगरानी करें और प्रबंधन प्रथाओं को समायोजित करें।")
        hinglish_suggestions.append("Kuch mausam tanav ka pata chala hai. Nazdik se nigrani karen aur management practices ko adjust karen.")
    else:
        suggestions.append("High weather stress detected. Take necessary measures such as increased irrigation and shading.")
        tamil_suggestions.append("அதிக வானிலை அழுத்தம் கண்டறியப்பட்டது. அதிகரித்த நீர்ப்பாசனம் மற்றும் நிழல் போன்ற தேவையான நடவடிக்கைகளை எடுக்கவும்.")
        hindi_suggestions.append("उच्च मौसम तनाव का पता चला है। आवश्यक उपाय जैसे कि सिंचाई बढ़ाना और छायांकन लेना।")
        hinglish_suggestions.append("High mausam tanav ka pata chala hai. Aavashyak upay jaise irrigation badhana aur shading lena.")

    return suggestions, tamil_suggestions,hindi_suggestions, hinglish_suggestions
    pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve user input from the form
            input_values = {
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'ph': float(request.form['ph']),
                'rainfall': float(request.form['rainfall'])
            }
            
            # Create DataFrame with proper feature names
            input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)
            
            # Scale the features using the scaler
            scaled_input = mx.transform(input_df)
            
            # Make prediction
            prediction = model.predict(scaled_input)
            predicted_value = prediction[0]
            
            # Convert prediction value to crop name
            predicted_crop = reverse_crop_dict.get(predicted_value, "Unknown Crop")
            
            # Calculate WSI
            wsi = calculate_wsi(**input_values)
            
            # Generate suggestions
            suggestions, tamil_suggestions, hindi_suggestions, hinglish_suggestions = generate_suggestions(
                input_values['N'], input_values['P'], input_values['K'],
                input_values['temperature'], input_values['humidity'],
                input_values['ph'], input_values['rainfall'], wsi
            )
            
            return render_template('index.html',
                                 predicted_crop=predicted_crop,
                                 wsi=round(wsi, 2),
                                 suggestions=suggestions,
                                 tamil_suggestions=tamil_suggestions,
                                 hindi_suggestions=hindi_suggestions,
                                 hinglish_suggestions=hinglish_suggestions)
                                 
        except Exception as e:
            # Handle any errors that might occur during prediction
            return render_template('index.html', 
                                 error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)