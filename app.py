import mysql.connector 
from flask import Flask, request, jsonify, render_template, request, redirect,flash, session
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# MySQL connection setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="legalassist"
)
cur = conn.cursor()

app.secret_key = 'your_secret_key_here'  

# Load dataset and clean missing values
df = pd.read_csv("ipc_sections.csv", encoding='latin-1')
df = df.dropna(subset=["Offense"])  # Remove rows with missing Offense

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df["Offense"] = df["Offense"].astype(str).apply(preprocess_text)
df["Description"] = df["Description"].astype(str)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["Offense"])

# Function to Extract Important Keywords
def extract_keywords(user_input, top_n=5):
    processed_input = preprocess_text(user_input)
    input_tfidf = vectorizer.transform([processed_input])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(input_tfidf.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return top_keywords

# Function to Predict Relevant IPC Sections Using Multiple Keywords
def predict_relevant_sections(user_input):
    keywords = extract_keywords(user_input, top_n=5)
    matched_sections = df[df["Offense"].apply(lambda x: any(keyword in x for keyword in keywords))]
    return matched_sections[["Section", "Description"]].drop_duplicates().head(30).to_dict(orient='records')

@app.route('/')
def lome():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Collect form data
        name = request.form['name']
        gender = request.form['gender']
        mobile = request.form['mobile']
        dob = request.form['dob']
        house_no = request.form['house_no']
        country = request.form['country']
        street = request.form['street']
        state = request.form['state']
        district = request.form['district']
        city = request.form['city']
        police_station = request.form['police_station']
        tehsil = request.form['tehsil']
        pincode = request.form['pincode']
        login_id = request.form['login_id']
        password = request.form['password']

        # Check if login_id already exists
        cur.execute("SELECT * FROM citizens WHERE login_id = %s", (login_id,))
        existing_user = cur.fetchone()

        if existing_user:
            return render_template('register.html', error="This Login ID already exists. Please use a different one.")

        # Insert into database
        cur.execute('''INSERT INTO citizens 
            (name, gender, mobile, dob, house_no, country, street, state, district, city, police_station, tehsil, pincode, login_id, password)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
            (name, gender, mobile, dob, house_no, country, street, state, district, city, police_station, tehsil, pincode, login_id, password)
        )
        conn.commit()

        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form['login_id']
        password = request.form['password']

        # Check user credentials
        cur.execute("SELECT * FROM citizens WHERE login_id = %s AND password = %s", (login_id, password))
        user = cur.fetchone()

        if user:
            return redirect('/option')  # Or your dashboard page
        else:
            return render_template('login.html', error="Invalid Login ID or Password")
    
    return render_template('login.html')

@app.route('/option')
def option():
    return render_template('option.html')
@app.route('/complaint', methods=['GET', 'POST'])
def complaint():
    if request.method == 'POST':
        fir_no = request.form['fir_no']
        police_station = request.form['police_station']
        district = request.form['district']
        station_no = request.form['station_no']
        occurrence_datetime = request.form['occurrence_datetime']
        informer_details = request.form['informer_details']
        place_of_occurrence = request.form['place_of_occurrence']
        criminal_details = request.form['criminal_details']
        investigation_steps = request.form['investigation_steps']
        despatch_datetime = request.form['despatch_datetime']
        designation = request.form['designation']

        if not all([fir_no, police_station, district, station_no, occurrence_datetime,
                    informer_details, place_of_occurrence, criminal_details,
                    investigation_steps, despatch_datetime, designation]):
            flash("⚠️ Please fill in all the details.")
            return render_template('complaint.html')
        cur.execute('''
            INSERT INTO complaints (
                fir_no, police_station, district, station_no, 
                occurrence_datetime, informer_details, place_of_occurrence, 
                criminal_details, investigation_steps, despatch_datetime, designation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            fir_no, police_station, district, station_no, 
            occurrence_datetime, informer_details, place_of_occurrence, 
            criminal_details, investigation_steps, despatch_datetime, designation
        ))
        conn.commit()
        flash("✅ Complaint Submitted Successfully!")
        return redirect('/option')
    return render_template('complaint.html')

# Home Route
@app.route('/LegalAssist')
def home():
    return render_template('LegalAssist.html')

# API Endpoint to Process Complaint and Return IPC Sections
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get('complaint', '')
    results = predict_relevant_sections(user_input)
    return jsonify(results)

@app.route('/fir/<fir_no>')
def fir_display(fir_no):
    cur.execute("SELECT * FROM complaints WHERE fir_no = %s", (fir_no,))
    complaint = cur.fetchone()

    if not complaint:
        flash("❌ FIR not found.")
        return redirect('/option')

    # Map columns from the table to a dictionary to pass to the template
    complaint_data = {
        'fir_no': complaint[0],
        'police_station': complaint[1],
        'district': complaint[2],
        'station_no': complaint[3],
        'occurrence_datetime': complaint[4],
        'informer_details': complaint[5],
        'place_of_occurrence': complaint[6],
        'criminal_details': complaint[7],
        'investigation_steps': complaint[8],
        'despatch_datetime': complaint[9],
        'designation': complaint[10]
    }

    return render_template('fir.html', data=complaint_data)



if __name__ == '__main__':
    app.run(debug=True)
