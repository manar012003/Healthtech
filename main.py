from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import numpy as np
import pandas as pd
import pickle
import sqlite3
import os
import requests 

from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
app.secret_key = 'supersecretkey'
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
OPENCAGE_KEY = os.getenv("OPENCAGE_KEY") 

# Load datasets
sym_des = pd.read_csv("ML/symtoms_df.csv")
precautions = pd.read_csv("ML/precautions_df.csv")
workout = pd.read_csv("ML/workout_df.csv")
description = pd.read_csv("ML/description.csv")
medications = pd.read_csv('ML/medications.csv')
diets = pd.read_csv("ML/diets.csv")

# Load model
svc = pickle.load(open('ML/svc.pkl','rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Symptom and disease mapping (keep your existing dicts)
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 
        'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 
        'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
         'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 
         'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 
         'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
          'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
          'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 
          'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
           'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
            'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 
            'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 
            'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 
            'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
             'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 
             'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
              'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76,
               'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 
               'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 
               'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 
               'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 
               'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 
               'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 
               'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 
               'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 
               'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
                'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 
                'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125,
                  'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 
                  'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 
'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 
'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis'
' (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 
40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 
'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 
18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 
'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 
'Psoriasis', 27: 'Impetigo'}

serious_keywords = ['heart', 'Allergy', 'Chronic cholestasis','Diabetes' ,'Fungal infection']
# ... (symptoms_dict and diseases_list stay the same)

# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
            
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            predicted_disease TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

#

def get_user_coordinates():
    try:
        ip_info = requests.get("https://ipinfo.io").json()
        loc = ip_info.get("loc")  # Format: "lat,lon"
        if loc:
            lat, lng = map(float, loc.split(","))
            print(f"[INFO] Got user location from IP: {lat}, {lng}")
            return lat, lng
    except Exception as e:
        print(f"[WARN] Failed to get IP location: {e}")
    
    # Default to Istanbul
    print("[INFO] Falling back to default coordinates: Istanbul")
    return 41.015137, 28.979530


import math
import requests  # Don't forget to import requests

def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # in meters

def get_nearest_hospital_overpass(lat, lng, radius=10000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
[out:json];
(
  node["amenity"="hospital"](around:{radius},{lat},{lng});
  way["amenity"="hospital"](around:{radius},{lat},{lng});
  relation["amenity"="hospital"](around:{radius},{lat},{lng});
);
out center;
"""

    print(f"Using lat={lat}, lng={lng}")

    try:
        response = requests.get(overpass_url, params={'data': query}, timeout=10)
        response.raise_for_status()
        data = response.json()

        hospitals = []
        for el in data.get('elements', []):
            tags = el.get('tags', {})
            name = tags.get('name')
            center = el.get('center') or el  # center for way/relation, node otherwise
            if name and 'lat' in center and 'lon' in center:
                distance = haversine(lat, lng, center['lat'], center['lon'])
                hospitals.append((distance, name))

        if hospitals:
            hospitals.sort()
            return f"{hospitals[0][1]} ({int(hospitals[0][0])} meters away)"
        else:
            return "No nearby named hospitals found."
    except Exception as e:
        print(f"[ERROR] Overpass API error: {e}")
        return "Could not retrieve hospital data."







# Routes
@app.route('/')
def index():
    if 'username' in session:
        username = session.get('username')
        return render_template('index.html', username=username)
    return redirect(url_for('login'))

  

@app.route('/profile')
def profile():
    username = session.get('username')
    if not username:
        return "Not logged in"

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT symptoms, predicted_disease FROM history
        WHERE username = ?
        ORDER BY id DESC
    ''', (username,))
    history = c.fetchall()
    conn.close()

    user_description = f"{username}'s profile - Welcome to your dashboard!"

    return render_template('profile.html', username=username, description=user_description, history=history)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    username = session.get('username')
    if not username:
        return "Not logged in", 403

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('DELETE FROM history WHERE username = ?', (username,))
    conn.commit()
    conn.close()

    return redirect(url_for('profile'))

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        symptoms = []
        for i in range(1, 5):
            symptom = request.form.get(f'symptom{i}')
            if symptom:
                symptoms.append(symptom.strip())

        if len(symptoms) < 4:
            message = "Please fill all four symptom fields."
            return render_template('index.html', message=message)

        try:
            predicted_disease = get_predicted_value(symptoms)

            # Store in DB
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO history (username, symptoms, predicted_disease)
                VALUES (?, ?, ?)
            ''', (session['username'], ", ".join(symptoms), predicted_disease))
            conn.commit()
            conn.close()

            if not predicted_disease:
                message = "Sorry, we couldn't predict a disease with the provided symptoms."
                return render_template('index.html', message=message)

            # Check if it's serious
            serious_keywords = ['heart', 'cancer', 'coma', 'Fungal infection']
            is_serious = any(keyword.lower() in predicted_disease.lower() for keyword in serious_keywords)

            nearest_hospital = None  # Default value to avoid UnboundLocalError

            if is_serious:
                lat = session.get('lat')
                lng = session.get('lng')
                if lat and lng:
                    nearest_hospital = get_nearest_hospital_overpass(lat, lng)
                else:
                    lat, lng = get_user_coordinates()
                    nearest_hospital = get_nearest_hospital_overpass(lat, lng)

            # Load extra info
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            my_precautions = [i for i in precautions[0]]

            return render_template(
                'index.html',
                predicted_disease=predicted_disease,
                dis_des=dis_des,
                my_precautions=my_precautions,
                medications=medications,
                my_diet=rec_diet,
                workout=workout,
                is_serious=is_serious,
                nearest_hospital=nearest_hospital  # will be None if not serious
            )


        except Exception as e:
            import traceback
            traceback.print_exc()
            message = f"An error occurred: {e}"
            return render_template('index.html', message=message)

    return render_template('index.html')


@app.route("/set_location", methods=["POST"])
def set_location():
    data = request.get_json()
    session["lat"] = data.get("lat")
    session["lng"] = data.get("lng")
    return "", 204


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash('Registration is done.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('This username is already taken', 'danger')
        finally:
            conn.close()
    return render_template('signup.html')


# Giriş yap
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Admin login check
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['username'] = username
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))

        # Normal user login check
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = username
            session['is_admin'] = False
            return redirect(url_for('index'))
        else:
            flash('Username or password is wrong', 'danger')

    return render_template('login.html')

  



# admin
@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or not session.get('is_admin'):
        flash('Access denied', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username FROM users")
    users = c.fetchall()
    conn.close()

    return render_template('admin.html', users=users)

# logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You logged out', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)



