import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import bcrypt
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import re
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
import speech_recognition as sr

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
jwtObject= JWTManager(app)

# Gemini API Configuration
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Updated System Prompt for Psychologist-Like Responses
MENTAL_HEALTH_PROMPT = """
You are MindEase, a compassionate mental health support companion designed to respond like a skilled psychologist. Your goal is to provide empathetic, supportive, and highly personalized responses that promote emotional well-being through counseling, motivational, and calming tones. Always be warm, understanding, and non-judgmental, using reflective listening to validate the user's emotions, motivational language to inspire resilience and small steps forward, and calming language to soothe and ground them. Respond in the user's preferred language (English or Malayalam) as specified. Do not diagnose conditions or provide medical advice. Keep responses concise (2-3 sentences) and deeply relevant to the user's detected mood and message.
"""

# Define distress keywords for triggering alerts, including new phrases
DISTRESS_KEYWORDS = [
    # Existing keywords
    "help", "crisis", "emergency", "urgent", "danger", "hurt", "scared", 
    "panic", "desperate", "suicide", "harm", "alone", "trapped", "overwhelmed",
    # New keywords for severe emotional distress
    "i want to die", "i don’t want to live anymore", "i’m done with life", 
    "i can’t go on", "i want to give up", "i’m thinking of killing myself", 
    "i’m going to end it all", "i don’t see a way out", "i’m planning to end my life",
    "there’s no point anymore", "i feel completely broken", "i’ve lost all hope", 
    "nothing matters anymore", "i’m a burden to everyone", "i can’t take it anymore", 
    "i feel empty inside", "i’m at my breaking point", "i’m drowning in my thoughts",
    "i have no one left", "i’m all alone in this", "no one cares about me", 
    "i’m better off gone", "i just want to disappear", "i feel invisible", 
    "i’m shutting everyone out", "i want to hurt myself", "i’ve been thinking about cutting",
    "i deserve to feel pain", "i need to make this pain stop", "i’ve been thinking about overdosing"
]

# Custom JWT error handlers for debugging
@jwtObject.invalid_token_loader
def invalid_token_callback(error):
    print("🔥 JWT Error: Invalid token", str(error))
    return jsonify({"status": "error", "message": "Invalid token", "error": str(error)}), 401

@jwtObject.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    print("🔥 JWT Error: Token has expired", jwt_payload)
    return jsonify({"status": "error", "message": "Token has expired", "error": "The token has expired"}), 401

@jwtObject.unauthorized_loader
def unauthorized_callback(error):
    print("🔥 JWT Error: Missing Authorization Header", str(error))
    return jsonify({"status": "error", "message": "Missing Authorization Header", "error": str(error)}), 401

# Email Configuration
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

# Function to Establish MySQL Connection
def get_db_connection():
    print("🔍 Connecting to database...")
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            port=3306,
            use_pure=True
        )
        print("✅ Database connection successful")
        return conn
    except Exception as e:
        print("🔥 Database Connection Error:", str(e))
        raise e

def decode_token(token):
    try:
        decoded = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
        return decoded
    except ExpiredSignatureError:
        raise Exception("Token has expired")
    except InvalidTokenError:
        raise Exception("Invalid token")

# Initialize sentiment analysis models
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update({
    "aching": -2.0,
    "sleepy": -0.5
})
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Audio Processing Functions
def save_audio(audio_samples, temp_file="temp_audio.wav"):
    try:
        print("Saving audio file...")
        # Ensure audio_samples is a NumPy array
        audio_samples = np.array(audio_samples, dtype=np.float32)
        # Normalize audio samples to prevent clipping
        if np.max(np.abs(audio_samples)) > 0:
            audio_samples = audio_samples / np.max(np.abs(audio_samples))
        sf.write(temp_file, audio_samples, 16000)
        print(f"Audio saved to {temp_file}")
        return temp_file
    except Exception as e:
        print(f"Error in save_audio: {str(e)}")
        return None

def transcribe_audio_to_text(audio_file_path, language='en-US', max_attempts=2):
    try:
        if not audio_file_path or not os.path.exists(audio_file_path):
            raise FileNotFoundError("Audio file not found for transcription")
        
        print(f"Transcribing audio to text in language: {language}...")
        recognizer = sr.Recognizer()
        attempt = 1
        while attempt <= max_attempts:
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = recognizer.record(source, duration=10)
                    text = recognizer.recognize_google(audio_data, language=language)
                    print(f"Transcribed text: {text}")
                    return text
            except sr.UnknownValueError:
                print(f"Attempt {attempt}: Speech Recognition could not understand audio")
                if attempt == max_attempts:
                    return "[Unrecognized speech after multiple attempts]"
                attempt += 1
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return "[Transcription error]"
    except Exception as e:
        print(f"Error in transcribe_audio_to_text: {str(e)}")
        return "[Transcription failed]"

# Function to Send Emergency Alerts to Contacts
def send_emergency_alert(user_id, session_id, alert_type, user_message):
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        # Fetch emergency contacts
        cursor.execute("SELECT contact_name, email FROM emergency_contact WHERE user_id = %s AND email IS NOT NULL", (user_id,))
        contacts = cursor.fetchall()
        if not contacts:
            print("⚠️ No emergency contacts with email found for user_id:", user_id)
            return

        # Fetch user's name
        cursor.execute("SELECT full_name FROM user WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            print("⚠️ User not found for user_id:", user_id)
            return
        user_name = user["full_name"]

        # Insert alert into emergency_alert table
        alert_message = f"{user_name} has sent a distress signal and may need your support. Their message: {user_message}"
        cursor.execute(
            "INSERT INTO emergency_alert (user_id, session_id, message, status) VALUES (%s, %s, %s, 'triggered')",
            (user_id, session_id, alert_message)
        )
        db.commit()

        # Send email to each contact
        for contact in contacts:
            if contact["email"]:
                msg = MIMEText(
                    f"Dear {contact['contact_name']},\n\n"
                    f"We wanted to let you know that {user_name} has sent a distress signal and might need your support. "
                    f"They shared: '{user_message}'.\n\n"
                    f"Please reach out to them when you can.\n\n"
                    f"Thank you,\nMindEase Team"
                )
                msg['Subject'] = f'MindEase Alert: {user_name} Needs Support'
                msg['From'] = EMAIL_ADDRESS
                msg['To'] = contact["email"]

                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    server.starttls()
                    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    server.sendmail(EMAIL_ADDRESS, [contact["email"]], msg.as_string())
                print(f"📧 Alert email sent to {contact['email']}")

        cursor.close()
        db.close()

    except Exception as e:
        print("🔥 ERROR in send_emergency_alert:", str(e))

# Home Route
@app.route("/")
def home():
    print("🏠 Home route accessed")
    return "Flask API is running!"

# API to Register a User
@app.route('/register', methods=['POST'])
def register():
    print("📝 Register endpoint called")
    try:
        db = get_db_connection()
        cursor = db.cursor()

        data = request.json
        full_name = data.get("fullName", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()
        confirm_password = data.get("confirmPassword", "").strip()
        date_of_birth = data.get("dateOfBirth", "").strip()
        gender = data.get("gender", "").strip()
        phone_number = data.get("phoneNumber", "").strip()

        if not all([full_name, email, password, confirm_password, date_of_birth, gender, phone_number]):
            return jsonify({"status": "error", "message": "All fields are required"}), 400

        if password != confirm_password:
            return jsonify({"status": "error", "message": "Passwords do not match"}), 400

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({"status": "error", "message": "Invalid email format"}), 400

        if len(password) < 8:
            return jsonify({"status": "error", "message": "Password must be at least 8 characters long"}), 400

        cursor.execute("SELECT id FROM user WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({"status": "error", "message": "Email already registered"}), 400

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        query = """
            INSERT INTO user (full_name, email, password, date_of_birth, gender, phone_number)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (full_name, email, hashed_password, date_of_birth, gender, phone_number))
        db.commit()

        user_id = cursor.lastrowid
        access_token = create_access_token(identity=str(user_id), expires_delta=timedelta(days=1))

        cursor.close()
        db.close()

        print(f"✅ Registration successful for user_id: {user_id}")
        return jsonify({
            "status": "success",
            "data": {
                "user_id": user_id,
                "access_token": access_token,
                "message": "User registered successfully"
            }
        })

    except Exception as e:
        print("🔥 ERROR in register:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Login
@app.route('/login', methods=['POST'])
def login():
    print("🔐 Login endpoint called")
    try:
        db = get_db_connection()
        cursor = db.cursor()

        data = request.json
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"status": "error", "message": "Email and password are required"}), 400

        cursor.execute("SELECT id, password FROM user WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            return jsonify({"status": "error", "message": "Invalid email or password"}), 401

        user_id, hashed_password = user

        if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            return jsonify({"status": "error", "message": "Invalid email or password"}), 401

        access_token = create_access_token(identity=str(user_id), expires_delta=timedelta(days=1))

        cursor.close()
        db.close()

        print(f"✅ Login successful for user_id: {user_id}")
        return jsonify({"status": "success", "data": {"access_token": access_token, "user_id": user_id, "message": "Login successful"}})

    except Exception as e:
        print("🔥 ERROR in login:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Forgot Password
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    try:
        db = get_db_connection()
        cursor = db.cursor()

        if not request.is_json:
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400

        email = request.json.get("email")
        if not email or not isinstance(email, str) or not email.strip():
            return jsonify({"status": "error", "message": "Valid email is required"}), 400

        email = email.strip()
        cursor.execute("SELECT id FROM user WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            return jsonify({"status": "error", "message": "Email not found"}), 404

        user_id = user[0]
        reset_token = create_access_token(identity=str(user_id), expires_delta=timedelta(hours=1))

        expiry_time = datetime.now() + timedelta(hours=1)
        cursor.execute(
            "INSERT INTO password_reset (user_id, token, expiry) VALUES (%s, %s, %s)",
            (user_id, reset_token, expiry_time)
        )
        db.commit()

        reset_link = f"http://localhost:3000/forgot-password?token={reset_token}"
        msg = MIMEText(f"Click here to reset your password: {reset_link}")
        msg['Subject'] = 'MindEase Password Reset'
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, [email], msg.as_string())

        return jsonify({"status": "success", "message": "Reset link sent to email", "reset_link": reset_link})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cursor.close()
        db.close()

# API to Reset Password
@app.route('/reset-password', methods=['POST'])
def reset_password():
    print("🔑 Reset-password endpoint called")
    try:
        db = get_db_connection()
        cursor = db.cursor()

        if not request.is_json:
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400

        data = request.json
        token = data.get("token")
        new_password = data.get("new_password")
        confirm_password = data.get("confirm_password")

        if not all([token, new_password, confirm_password]):
            return jsonify({"status": "error", "message": "Token, new password, and confirm password are required"}), 400

        if not isinstance(new_password, str) or not isinstance(confirm_password, str):
            return jsonify({"status": "error", "message": "Passwords must be strings"}), 400

        if new_password != confirm_password:
            return jsonify({"status": "error", "message": "Passwords do not match"}), 400

        if len(new_password) < 8:
            return jsonify({"status": "error", "message": "Password must be at least 8 characters long"}), 400

        print(f"Validating token: {token}")
        try:
            decoded_token = decode_token(token)
            user_id = decoded_token['sub']
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid or expired token: {str(e)}"}), 401

        cursor.execute("SELECT expiry FROM password_reset WHERE user_id = %s AND token = %s", (user_id, token))
        row = cursor.fetchone()
        if not row:
            return jsonify({"status": "error", "message": "Invalid or expired token"}), 401

        expiry = row[0]
        if datetime.now() > expiry:
            return jsonify({"status": "error", "message": "Token has expired"}), 401

        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("UPDATE user SET password = %s WHERE id = %s", (hashed_password, user_id))
        cursor.execute("DELETE FROM password_reset WHERE user_id = %s", (user_id,))
        db.commit()

        print(f"✅ Password reset successful for user_id: {user_id}")
        return jsonify({"status": "success", "message": "Password reset successful"})

    except mysql.connector.Error as db_err:
        print(f"🔥 Database Error: {str(db_err)}")
        return jsonify({"status": "error", "message": f"Database error: {str(db_err)}"}), 500
    except Exception as e:
        print(f"🔥 General Error: {str(e)}")
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500
    finally:
        cursor.close()
        if 'db' in locals():
            db.close()

# API to Start a Session
@app.route('/start_session', methods=['POST'])
@jwt_required()
def start_session():
    print("🚀 Start-session endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        query = "INSERT INTO session (user_id, status) VALUES (%s, 'active')"
        cursor.execute(query, (user_id,))
        db.commit()

        session_id = cursor.lastrowid
        cursor.close()
        db.close()

        print("Session created with session_id:", session_id)
        return jsonify({"status": "success", "data": {"session_id": session_id, "user_id": user_id, "status": "active"}})

    except Exception as e:
        print("🔥 ERROR in start_session:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to End a Session
@app.route('/end_session/<int:session_id>', methods=['PUT'])
@jwt_required()
def end_session(session_id):
    print("⏹️ End-session endpoint called for session_id:", session_id)
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)

        db = get_db_connection()
        cursor = db.cursor()

        cursor.execute("SELECT user_id FROM session WHERE id = %s", (session_id,))
        session = cursor.fetchone()

        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 404

        if int(session[0]) != user_id:
            return jsonify({"status": "error", "message": "Unauthorized: session does not belong to the current user"}), 403

        query = "UPDATE session SET end_time = NOW(), status = 'closed' WHERE id = %s"
        cursor.execute(query, (session_id,))
        db.commit()

        cursor.close()
        db.close()

        print("✅ Session ended successfully")
        return jsonify({"status": "success", "data": {"session_id": session_id, "status": "closed"}})

    except Exception as e:
        print("🔥 ERROR in end_session:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Get All Sessions for a User
@app.route('/get_sessions', methods=['GET'])
@jwt_required()
def get_sessions():
    print("📋 Get-sessions endpoint called")
    try:
        user_id = get_jwt_identity()
        db = get_db_connection()
        cursor = db.cursor()

        query = "SELECT id, start_time, end_time, status FROM session WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        sessions = cursor.fetchall()

        session_list = [
            {"session_id": s[0], "start_time": str(s[1]), "end_time": str(s[2]) if s[2] else None, "status": s[3]}
            for s in sessions
        ]

        cursor.close()
        db.close()

        return jsonify({"status": "success", "data": {"sessions": session_list}})

    except Exception as e:
        print("🔥 ERROR in get_sessions:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Delete a Session
@app.route('/delete_session/<int:session_id>', methods=['DELETE'])
@jwt_required()
def delete_session(session_id):
    print("🗑️ Delete-session endpoint called for session_id:", session_id)
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        cursor.execute("SELECT user_id FROM session WHERE id = %s", (session_id,))
        session = cursor.fetchone()
        if not session or session[0] != user_id:
            return jsonify({"status": "error", "message": "Unauthorized or session not found"}), 403

        query = "DELETE FROM session WHERE id = %s"
        cursor.execute(query, (session_id,))
        db.commit()

        cursor.close()
        db.close()

        return jsonify({"status": "success", "message": f"Session {session_id} deleted successfully"})

    except Exception as e:
        print("🔥 ERROR in delete_session:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to View Profile
@app.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    current_user_id = get_jwt_identity()

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT full_name, email, gender, date_of_birth, phone_number FROM user WHERE id = %s", (current_user_id,))
        user = cursor.fetchone()
        cursor.close()
        db.close()

        if user:
            if user["date_of_birth"]:
                user["date_of_birth"] = user["date_of_birth"].strftime('%Y-%m-%d')
            return jsonify(user=user), 200
        else:
            return jsonify({"msg": "User not found"}), 404
    except Exception as e:
        print("DB error:", e)
        return jsonify({"msg": str(e)}), 500

# API to Get User
@app.route('/get_user', methods=['GET'])
@jwt_required()
def get_user():
    print("🔍 Get user endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        cursor.execute("SELECT full_name FROM user WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404

        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "full_name": user[0]
            }
        })

    except Exception as e:
        print("🔥 ERROR in get_user:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Clear Chats
@app.route("/clear_chats/<session_id>", methods=["DELETE"])
@jwt_required()
def clear_chats(session_id):
    print(f"🗑️ Clear-chats endpoint called for session_id: {session_id}")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        cursor.execute("SELECT user_id FROM session WHERE id = %s", (session_id,))
        session = cursor.fetchone()
        if not session or session[0] != user_id:
            return jsonify({"status": "error", "message": "Unauthorized or session not found"}), 403

        cursor.execute("DELETE FROM chat WHERE session_id = %s", (session_id,))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"status": "success", "message": "Chat history cleared"})
    except Exception as e:
        print("🔥 ERROR in clear_chats:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Add an Emergency Contact
@app.route('/add_emergency_contact', methods=['POST'])
@jwt_required()
def add_emergency_contact():
    print("📞 Add-emergency-contact endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        data = request.json
        contact_name = data.get("contact_name", "").strip()
        phone_number = data.get("phone_number", "").strip()
        email = data.get("email", "").strip()
        relationship = data.get("relationship", "").strip()

        # Validate required fields
        if not contact_name:
            return jsonify({"status": "error", "message": "Contact name cannot be empty"}), 400
        if len(contact_name) > 100:
            return jsonify({"status": "error", "message": "Contact name must be 100 characters or less"}), 400

        if not phone_number or not re.match(r"^[0-9]{10,15}$", phone_number):
            return jsonify({"status": "error", "message": "Phone number must be 10-15 digits"}), 400

        if not email:
            return jsonify({"status": "error", "message": "Email is required"}), 400
        if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
            return jsonify({"status": "error", "message": "Invalid email format"}), 400

        allowed_relationships = ['Family', 'Friend', 'Guardian', 'Other']
        if not relationship or relationship not in allowed_relationships:
            return jsonify({"status": "error", "message": "Relationship must be one of: Family, Friend, Guardian, Other"}), 400

        # Check maximum contacts limit (e.g., 5)
        cursor.execute("SELECT COUNT(*) FROM emergency_contact WHERE user_id = %s", (user_id,))
        contact_count = cursor.fetchone()[0]
        if contact_count >= 5:
            return jsonify({"status": "error", "message": "Maximum number of emergency contacts reached (5)"}), 400

        # Insert the emergency contact
        query = """
            INSERT INTO emergency_contact (user_id, contact_name, phone_number, email, relationship)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (user_id, contact_name, phone_number, email, relationship))
        db.commit()

        contact_id = cursor.lastrowid
        cursor.close()
        db.close()

        print(f"✅ Emergency contact added for user_id: {user_id}, contact_id: {contact_id}")
        return jsonify({
            "status": "success",
            "data": {
                "contact_id": contact_id,
                "message": "Emergency contact added successfully"
            }
        })

    except mysql.connector.Error as db_err:
        print(f"🔥 Database Error in add_emergency_contact: {str(db_err)}")
        return jsonify({"status": "error", "message": f"Database error: {str(db_err)}"}), 500
    except Exception as e:
        print("🔥 ERROR in add_emergency_contact:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Get All Emergency Contacts for a User
@app.route('/get_emergency_contacts', methods=['GET'])
@jwt_required()
def get_emergency_contacts():
    print("📞 Get-emergency-contacts endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        query = """
            SELECT id, contact_name, phone_number, email, relationship, created_at
            FROM emergency_contact
            WHERE user_id = %s
            ORDER BY created_at DESC
        """
        cursor.execute(query, (user_id,))
        contacts = cursor.fetchall()

        contact_list = [
            {
                "id": contact["id"],
                "contact_name": contact["contact_name"],
                "phone_number": contact["phone_number"],
                "email": contact["email"],
                "relationship": contact["relationship"],
                "created_at": str(contact["created_at"])
            }
            for contact in contacts
        ]

        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "contacts": contact_list
            }
        })

    except Exception as e:
        print("🔥 ERROR in get_emergency_contacts:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Update an Emergency Contact
@app.route('/update_emergency_contact/<int:contact_id>', methods=['PUT'])
@jwt_required()
def update_emergency_contact(contact_id):
    print(f"📞 Update-emergency-contact endpoint called for contact_id: {contact_id}")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        # Check if the contact exists and belongs to the user
        cursor.execute("SELECT user_id FROM emergency_contact WHERE id = %s", (contact_id,))
        contact = cursor.fetchone()
        if not contact:
            return jsonify({"status": "error", "message": "Contact not found"}), 404
        if contact[0] != user_id:
            return jsonify({"status": "error", "message": "Unauthorized: Contact does not belong to the current user"}), 403

        data = request.json
        contact_name = data.get("contact_name", "").strip()
        phone_number = data.get("phone_number", "").strip()
        email = data.get("email", "").strip()
        relationship = data.get("relationship", "").strip()

        # Validate required fields
        if not contact_name:
            return jsonify({"status": "error", "message": "Contact name cannot be empty"}), 400
        if len(contact_name) > 100:
            return jsonify({"status": "error", "message": "Contact name must be 100 characters or less"}), 400

        if not phone_number or not re.match(r"^[0-9]{10,15}$", phone_number):
            return jsonify({"status": "error", "message": "Phone number must be 10-15 digits"}), 400

        if not email:
            return jsonify({"status": "error", "message": "Email is required"}), 400
        if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
            return jsonify({"status": "error", "message": "Invalid email format"}), 400

        allowed_relationships = ['Family', 'Friend', 'Guardian', 'Other']
        if not relationship or relationship not in allowed_relationships:
            return jsonify({"status": "error", "message": "Relationship must be one of: Family, Friend, Guardian, Other"}), 400

        # Update the emergency contact
        query = """
            UPDATE emergency_contact
            SET contact_name = %s, phone_number = %s, email = %s, relationship = %s
            WHERE id = %s AND user_id = %s
        """
        cursor.execute(query, (contact_name, phone_number, email, relationship, contact_id, user_id))
        db.commit()

        if cursor.rowcount == 0:
            return jsonify({"status": "error", "message": "No changes made to the contact"}), 400

        cursor.close()
        db.close()

        print(f"✅ Emergency contact updated for user_id: {user_id}, contact_id: {contact_id}")
        return jsonify({
            "status": "success",
            "data": {
                "contact_id": contact_id,
                "message": "Emergency contact updated successfully"
            }
        })

    except mysql.connector.Error as db_err:
        print(f"🔥 Database Error in update_emergency_contact: {str(db_err)}")
        return jsonify({"status": "error", "message": f"Database error: {str(db_err)}"}), 500
    except Exception as e:
        print("🔥 ERROR in update_emergency_contact:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Delete an Emergency Contact
@app.route('/delete_emergency_contact/<int:contact_id>', methods=['DELETE'])
@jwt_required()
def delete_emergency_contact(contact_id):
    print(f"📞 Delete-emergency-contact endpoint called for contact_id: {contact_id}")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        # Check if the contact exists and belongs to the user
        cursor.execute("SELECT user_id FROM emergency_contact WHERE id = %s", (contact_id,))
        contact = cursor.fetchone()
        if not contact:
            return jsonify({"status": "error", "message": "Contact not found"}), 404
        if contact[0] != user_id:
            return jsonify({"status": "error", "message": "Unauthorized: Contact does not belong to the current user"}), 403

        # Check minimum contacts requirement
        cursor.execute("SELECT COUNT(*) FROM emergency_contact WHERE user_id = %s", (user_id,))
        contact_count = cursor.fetchone()[0]
        if contact_count <= 2:
            return jsonify({"status": "error", "message": "Cannot delete contact: At least 2 emergency contacts are required"}), 400

        # Delete the emergency contact
        cursor.execute("DELETE FROM emergency_contact WHERE id = %s AND user_id = %s", (contact_id, user_id))
        db.commit()

        if cursor.rowcount == 0:
            return jsonify({"status": "error", "message": "No contact was deleted"}), 400

        cursor.close()
        db.close()

        print(f"✅ Emergency contact deleted for user_id: {user_id}, contact_id: {contact_id}")
        return jsonify({
            "status": "success",
            "message": f"Emergency contact {contact_id} deleted successfully"
        })

    except Exception as e:
        print("🔥 ERROR in delete_emergency_contact:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Get Emergency Alerts for a User
@app.route('/get_emergency_alerts', methods=['GET'])
@jwt_required()
def get_emergency_alerts():
    print("🚨 Get-emergency-alerts endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        query = """
            SELECT id, session_id, message, timestamp, status
            FROM emergency_alert
            WHERE user_id = %s
            ORDER BY timestamp DESC
        """
        cursor.execute(query, (user_id,))
        alerts = cursor.fetchall()

        alert_list = [
            {
                "id": alert["id"],
                "session_id": alert["session_id"],
                "message": alert["message"],
                "timestamp": str(alert["timestamp"]),
                "status": alert["status"]
            }
            for alert in alerts
        ]

        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "alerts": alert_list
            }
        })

    except Exception as e:
        print("🔥 ERROR in get_emergency_alerts:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500
      # Feedback submission endpoint
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        print(f"Received feedback data: {data}")  # Log the incoming data for debugging
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')
        rating = data.get('rating')

        # Validate input
        if not all([name, email, message]) or rating is None:
            return jsonify({"status": "error", "message": "All fields are required"}), 400
        
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({"status": "error", "message": "Rating must be an integer between 1 and 5"}), 400

        # Insert feedback into the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (name, email, message, rating)
            VALUES (%s, %s, %s, %s)
        ''', (name, email, message, rating))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Feedback submitted successfully"}), 200

    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return jsonify({"status": "error", "message": "Failed to submit feedback"}), 500
        
@app.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_message():
    print("🔍 Analyze endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        session_id = None
        user_message = ""
        conversation_history = []
        language = 'en-US'  # Default language
        audio_blob = request.files.get('audio')

        if audio_blob:
            print("Audio input detected")
            data = request.form
            session_id = data.get("session_id")
            conversation_history = json.loads(data.get("conversation_history", "[]"))
            language = data.get("language", "en-US")  # Get language from frontend
        else:
            print("Text input detected")
            data = request.json or {}
            session_id = data.get("session_id")
            user_message = data.get("message", "").strip()
            conversation_history = data.get("conversation_history", [])
            language = data.get("language", "en-US")  # Get language from frontend

        if not session_id:
            print("🔥 Error: Session ID is required")
            return jsonify({"status": "error", "message": "Session ID is required"}), 400

        if not user_message and not audio_blob:
            print("🔥 Error: Message or audio is required")
            return jsonify({"status": "error", "message": "Message or audio is required"}), 400

        cursor.execute("SELECT user_id FROM session WHERE id = %s", (session_id,))
        session = cursor.fetchone()
        if not session or session[0] != user_id:
            print("🔥 Error: Unauthorized or session not found")
            return jsonify({"status": "error", "message": "Unauthorized or session not found"}), 403

        mood_label = "Neutral 🙂"
        bot_response = ""
        vader_result = {'compound': 0.0}
        emotion_result = {'label': 'neutral', 'score': 0.0}
        trigger_alert = False

        if audio_blob:
            audio_data = BytesIO(audio_blob.read())
            audio_data.seek(0, os.SEEK_END)
            audio_size = audio_data.tell()
            print(f"Audio blob size: {audio_size} bytes")
            if audio_size < 100:
                print("🔥 Error: Audio blob is too small or empty")
                user_message = "[Voice input: Invalid audio]"
                bot_response = "I’m sorry, your voice message seems to be empty or too short. Could you try recording again for a few seconds, or type your message instead?"
                return jsonify({
                    "status": "success",
                    "data": {
                        "chat_id": 0,
                        "message": "Voice message sent 🎙️",
                        "transcribed_text": user_message,
                        "response": bot_response,
                        "mood_label": mood_label,
                        "self_help": [{"title": "No resources available at the moment.", "link": "#"}],
                        "mood_tracked": False
                    }
                }), 200

            try:
                audio_data.seek(0)
                audio_samples, sr = librosa.load(audio_data, sr=16000, mono=True)
                audio_duration = len(audio_samples) / sr
                print(f"Audio loaded: duration={audio_duration:.2f}s, sample_rate={sr}, samples={len(audio_samples)}")
                if len(audio_samples) == 0:
                    raise ValueError("Audio samples are empty")
            except Exception as e:
                print(f"🔥 Error loading audio with librosa: {str(e)}")
                user_message = "[Voice input: Unable to process audio]"
                bot_response = "I’m sorry, I couldn’t process your voice message due to an audio issue. Could you try recording again or typing your message instead?"
                return jsonify({
                    "status": "success",
                    "data": {
                        "chat_id": 0,
                        "message": "Voice message sent 🎙️",
                        "transcribed_text": user_message,
                        "response": bot_response,
                        "mood_label": mood_label,
                        "self_help": [{"title": "No resources available at the moment.", "link": "#"}],
                        "mood_tracked": False
                    }
                }), 200

            temp_file = save_audio(audio_samples, "temp_audio.wav")
            if not temp_file:
                print("🔥 Error: Failed to save audio file")
                user_message = "[Voice input: Unable to process audio]"
                bot_response = "I’m sorry, I couldn’t process your voice message due to an issue saving the audio. Could you try recording again or typing your message instead?"
                return jsonify({
                    "status": "success",
                    "data": {
                        "chat_id": 0,
                        "message": "Voice message sent 🎙️",
                        "transcribed_text": user_message,
                        "response": bot_response,
                        "mood_label": mood_label,
                        "self_help": [{"title": "No resources available at the moment.", "link": "#"}],
                        "mood_tracked": False
                    }
                }), 200

            try:
                transcription = transcribe_audio_to_text(temp_file, language=language, max_attempts=2)
                user_message = transcription if "[Transcription" not in transcription else "[Voice input: Unable to transcribe]"
                print(f"Final user message: {user_message}")
                if "[Transcription" in transcription:
                    bot_response = "I’m sorry, I couldn’t understand your voice message this time. Could you try speaking clearly or typing your message instead?"
            except Exception as e:
                print(f"🔥 Error during transcription: {str(e)}")
                user_message = "[Voice input: Transcription error]"
                bot_response = "I’m sorry, there was an issue transcribing your voice message. Could you try speaking clearly or typing your message instead?"
                return jsonify({
                    "status": "success",
                    "data": {
                        "chat_id": 0,
                        "message": "Voice message sent 🎙️",
                        "transcribed_text": user_message,
                        "response": bot_response,
                        "mood_label": mood_label,
                        "self_help": [{"title": "No resources available at the moment.", "link": "#"}],
                        "mood_tracked": False
                    }
                }), 200

            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                print("Temporary audio file removed")

        # Check for distress keywords and phrases in the user message
        if "[Transcription" not in user_message and "[Voice input" not in user_message:
            message_lower = user_message.lower()
            for keyword in DISTRESS_KEYWORDS:
                if keyword in message_lower:
                    trigger_alert = True
                    print(f"⚠️ Distress keyword/phrase detected: '{keyword}'")
                    break

        # Perform sentiment and emotion analysis
        if "[Transcription" not in user_message and "[Voice input" not in user_message:
            try:
                vader_result = analyzer.polarity_scores(user_message)
                emotion_result = emotion_pipeline(user_message)[0]
                vader_compound = vader_result.get("compound", 0.0)
                emotion_label = emotion_result.get("label", "neutral")
                emotion_score = emotion_result.get("score", 0.0)

                print(f"VADER compound: {vader_compound}, Emotion label: {emotion_label}, Emotion score: {emotion_score}")

                # Check for severe distress based on sentiment and emotion thresholds
                if not trigger_alert:  # Only check if keywords didn't already trigger
                    if vader_compound < -0.8 and emotion_label in ["sadness", "fear"] and emotion_score > 0.9:
                        trigger_alert = True
                        print("⚠️ Severe emotional distress detected based on sentiment and emotion scores")

                # Updated mood classification logic to prioritize greetings, "tired", "happy", and "excited"
                message_lower = user_message.lower()
                greetings = ["hi", "hello", "hey"]
                if any(greeting in message_lower for greeting in greetings):  # Explicitly check for greetings
                    mood_label = "Neutral 🙂"
                    print("Explicitly detected a greeting in message, setting mood to Neutral 🙂")
                elif "tired" in message_lower:  # Explicitly check for "tired" keyword
                    mood_label = "Tired 😴"
                    print("Explicitly detected 'tired' in message, setting mood to Tired 😴")
                elif "happy" in message_lower:  # Explicitly check for "happy" keyword
                    mood_label = "Happy 😊"
                    print("Explicitly detected 'happy' in message, setting mood to Happy 😊")
                elif "excited" in message_lower:  # Explicitly check for "excited" keyword
                    mood_label = "Excited 🎉"
                    print("Explicitly detected 'excited' in message, setting mood to Excited 🎉")
                elif emotion_label == "joy":
                    if vader_compound >= 0.5 and emotion_score > 0.9:
                        mood_label = "Excited 🎉"
                    else:
                        mood_label = "Happy 😊"
                elif emotion_label == "sadness":
                    if vader_compound < -0.5:
                        mood_label = "Sad 😔"
                    elif -0.5 <= vader_compound <= -0.1 and emotion_score <= 0.7:
                        mood_label = "Tired 😴"
                    else:
                        mood_label = "Sad 😔"
                elif emotion_label == "anger":
                    if vader_compound <= -0.5 and emotion_score > 0.9:
                        mood_label = "Angry 😡"
                    else:
                        mood_label = "Stressed 😟"
                elif emotion_label == "fear" or emotion_label == "disgust":
                    if emotion_score > 0.8:
                        mood_label = "Stressed 😟"
                    else:
                        if -0.1 < vader_compound < 0.3:
                            mood_label = "Neutral 🙂"
                        elif vader_compound <= -0.1:
                            mood_label = "Tired 😴" if -0.5 <= vader_compound <= -0.1 and emotion_score <= 0.7 else "Sad 😔"
                        else:
                            mood_label = "Happy 😊"
                elif emotion_label == "surprise":
                    mood_label = "Excited 🎉" if vader_compound >= 0.3 else "Neutral 🙂"
                else:
                    if -0.1 < vader_compound < 0.3:
                        mood_label = "Neutral 🙂"
                    elif vader_compound <= -0.1:
                        mood_label = "Tired 😴" if -0.5 <= vader_compound <= -0.1 and emotion_score <= 0.7 else "Sad 😔"
                    else:
                        mood_label = "Happy 😊"

            except Exception as e:
                print(f"🔥 Error in sentiment/emotion analysis: {str(e)}")
                mood_label = "Neutral 🙂"
        else:
            mood_label = "Neutral 🙂"

        print(f"Detected emotion: {mood_label}")

        # Trigger emergency alert if distress keywords or severe sentiment are detected
        if trigger_alert:
            print(f"⚠️ Triggering emergency alert due to distress detection in message: {user_message}")
            send_emergency_alert(user_id, session_id, "Distress", user_message)

        # Generate bot response
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history]) if conversation_history else "No previous conversation."
        sentiment_info = f"Sentiment (VADER): {vader_result.get('compound', 0.0)}, Emotion: {emotion_result.get('label', 'neutral')} (Confidence: {emotion_result.get('score', 0.0)}), Detected Mood: {mood_label}"
        
        prompt = f"""
        {MENTAL_HEALTH_PROMPT}
        Below is the conversation history between the user and the assistant.
        Conversation history:\n{history_text}\n
        Current user message: "{user_message}"\n
        {sentiment_info}\n
        Distress keywords detected: {trigger_alert}\n
        User language preference: {language} (respond in {'Malayalam' if language == 'ml-IN' else 'English'})\n

        Craft a **highly personalized, empathetic, and supportive** response that:
        - Reflects a psychologist-like tone, blending counseling, motivational, and calming approaches based on the detected mood ({mood_label}) and distress signals.
        - If distress keywords were detected, acknowledge the urgency gently (e.g., "It sounds like you’re going through something really tough right now") and suggest immediate support options (e.g., reaching out to a trusted contact).
        - Respond in the user's preferred language ({'Malayalam' if language == 'ml-IN' else 'English'}).
        - Do not repeat the user's exact message or specific details.
        - Keep the response concise (2-3 sentences).
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            chat = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
            bot_response = chat.send_message(prompt).text
            print(f"Gemini response: {bot_response}")
        except Exception as gemini_error:
            print(f"🔥 Gemini Error: {str(gemini_error)}")
            bot_response = "I’m sorry, I couldn’t generate a response due to a technical issue. Could you try again or type your message instead?" if language == 'en-US' else "ക്ഷമിക്കണം, സാങ്കേതിക പ്രശ്നം കാരണം എനിക്ക് പ്രതികരിക്കാൻ കഴിഞ്ഞില്ല. വീണ്ടും ശ്രമിക്കാമോ അല്ലെങ്കിൽ നിന്റെ സന്ദേശം ടൈപ്പ് ചെയ്യാമോ?"
            mood_label = "Neutral 🙂"

        # Fetch self-help resources
        try:
            cursor.execute("SELECT title, link FROM self_help_resources WHERE mood_label = %s", (mood_label,))
            resources = cursor.fetchall()
            self_help = [{"title": res[0], "link": res[1]} for res in resources] or [{"title": "No resources available at the moment.", "link": "#"}]
        except Exception as e:
            print(f"�fire Error fetching self-help resources: {str(e)}")
            self_help = [{"title": "No resources available at the moment.", "link": "#"}]

        # Store chat data
        try:
            cursor.execute("""
                INSERT INTO chat (user_id, session_id, message, response, vader_score, bert_label, bert_score, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (user_id, session_id, user_message, bot_response, 
                  vader_result.get('compound', 0.0), 
                  emotion_result.get('label', 'neutral'), 
                  emotion_result.get('score', 0.0)))
        except Exception as e:
            print(f"🔥 Error storing chat data: {str(e)}")
            bot_response = "I’m sorry, I couldn’t save our chat due to a database issue. Let’s try again—what’s on your mind?" if language == 'en-US' else "ക്ഷമിക്കണം, ഡാറ്റാബേസ് പ്രശ്നം കാരണം ഞങ്ങളുടെ ചാറ്റ് സേവ് ചെയ്യാൻ കഴിഞ്ഞില്ല. വീണ്ടും ശ്രമിക്കാം—നിന്റെ മനസ്സിൽ എന്താണ്?"
            mood_label = "Neutral 🙂"

        # Store mood data
        try:
            cursor.execute("""
                INSERT INTO mood (user_id, session_id, message, vader_score, bert_label, bert_score, mood_label, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (user_id, session_id, user_message, 
                  vader_result.get('compound', 0.0), 
                  emotion_result.get('label', 'neutral'), 
                  emotion_result.get('score', 0.0), mood_label))
        except Exception as e:
            print(f"🔥 Error storing mood data: {str(e)}")

        db.commit()
        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "chat_id": cursor.lastrowid if 'cursor' in locals() else 0,
                "message": "Voice message sent 🎙️" if audio_blob else user_message,
                "transcribed_text": user_message if audio_blob and "[Transcription" not in user_message and "[Voice input" not in user_message else None,
                "response": bot_response,
                "mood_label": mood_label,
                "self_help": self_help,
                "mood_tracked": True,
                "alert_triggered": trigger_alert
            }
        })

    except Exception as e:
        print("🔥 ERROR in analyze:", str(e))
        if 'db' in locals():
            cursor.close()
            db.close()
        return jsonify({
            "status": "success",
            "data": {
                "chat_id": 0,
                "message": "Voice message sent 🎙️" if 'audio_blob' in locals() and audio_blob else user_message if 'user_message' in locals() else "[Voice input: Unable to process]",
                "transcribed_text": user_message if 'user_message' in locals() and "[Transcription" not in user_message and "[Voice input" not in user_message else None,
                "response": "I’m sorry, something went wrong while processing your message. I’m still here for you—let’s try again. What’s on your mind?" if language == 'en-US' else "ക്ഷമിക്കണം, നിന്റെ സന്ദേശം പ്രോസസ്സ് ചെയ്യുന്നതിനിടയിൽ എന്തോ കുഴപ്പം സംഭവിച്ചു. ഞാൻ ഇപ്പോഴും ഇവിടെ ഉണ്ട്—വീണ്ടും ശ്രമിക്കാം. നിന്റെ മനസ്സിൽ എന്താണ്?",
                "mood_label": "Neutral 🙂",
                "self_help": [{"title": "No resources available at the moment.", "link": "#"}],
                "mood_tracked": False,
                "alert_triggered": False
            }
        }), 200

# API to Fetch Chat History by Session
@app.route("/get_chats/<int:session_id>", methods=["GET"])
@jwt_required()
def get_chats(session_id):
    print("💬 Get-chats endpoint called for session_id:", session_id)
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        cursor.execute("SELECT user_id FROM session WHERE id = %s", (session_id,))
        session = cursor.fetchone()
        print("Session user_id from DB:", session[0] if session else "None")
        if not session or session[0] != user_id:
            return jsonify({"status": "error", "message": "Unauthorized or session not found"}), 403

        query = "SELECT id, message, response, timestamp FROM chat WHERE session_id = %s ORDER BY timestamp DESC"
        cursor.execute(query, (session_id,))
        chats = cursor.fetchall()

        chat_list = [
            {
                "chat_id": chat[0],
                "message": chat[1],
                "response": chat[2],
                "timestamp": str(chat[3])
            }
            for chat in chats
        ]

        cursor.close()
        db.close()

        return jsonify({"status": "success", "data": {"chats": chat_list}})

    except Exception as e:
        print("🔥 ERROR in get_chats:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Fetch All Chat History for a User (Across All Sessions)
@app.route('/get_user_chat_history', methods=['GET'])
@jwt_required()
def get_user_chat_history():
    print("📜 Get-user-chat-history endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        # Step 1: Fetch all session IDs for the user
        cursor.execute("SELECT id FROM session WHERE user_id = %s ORDER BY start_time DESC", (user_id,))
        sessions = cursor.fetchall()
        if not sessions:
            cursor.close()
            db.close()
            return jsonify({
                "status": "success",
                "data": {
                    "chat_history": [],
                    "message": "No sessions found for this user"
                }
            }), 200

        session_ids = [session['id'] for session in sessions]

        # Step 2: Dynamically create placeholders for the IN clause
        placeholders = ','.join(['%s'] * len(session_ids))  # Creates '%s,%s,%s,...' based on the number of session IDs
        query = f"""
            SELECT c.id, c.session_id, c.message, c.response, c.timestamp, s.start_time
            FROM chat c
            JOIN session s ON c.session_id = s.id
            WHERE c.session_id IN ({placeholders})
            ORDER BY c.timestamp DESC
        """
        cursor.execute(query, session_ids)  # Pass the session_ids list directly

        chats = cursor.fetchall()

        # Step 3: Structure the chat history
        chat_history = [
            {
                "chat_id": chat["id"],
                "session_id": chat["session_id"],
                "session_start_time": str(chat["start_time"]),
                "message": chat["message"],
                "response": chat["response"],
                "timestamp": str(chat["timestamp"])
            }
            for chat in chats
        ]

        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "chat_history": chat_history,
                "message": f"Retrieved {len(chat_history)} chat entries for user {user_id}"
            }
        }), 200

    except Exception as e:
        print("🔥 ERROR in get_user_chat_history:", str(e))
        if 'db' in locals():
            cursor.close()
            db.close()
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Fetch Self-Help Resources Based on Latest Mood or Specific Mood
@app.route('/get_self_help', methods=['GET'])
@jwt_required()
def get_self_help():
    print("📚 Get-self-help endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        mood_label = request.args.get('mood_label', None)
        
        if not mood_label:
            cursor.execute(
                "SELECT mood_label FROM mood WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1",
                (user_id,)
            )
            mood_result = cursor.fetchone()
            mood_label = mood_result[0] if mood_result else "Neutral 🙂"
            print(f"Latest mood for user {user_id}: {mood_label}")
        else:
            print(f"Fetching resources for specified mood: {mood_label}")

        allowed_moods = ["Happy 😊", "Sad 😔", "Angry 😡", "Stressed 😟", "Tired 😴", "Excited 🎉", "Neutral 🙂"]
        if mood_label not in allowed_moods:
            print(f"Invalid mood_label provided: {mood_label}, defaulting to Neutral")
            mood_label = "Neutral 🙂"

        cursor.execute(
            "SELECT title, link FROM self_help_resources WHERE mood_label = %s",
            (mood_label,)
        )
        resources = cursor.fetchall()
        self_help = [{"title": res[0], "link": res[1]} for res in resources]

        if not self_help:
            print(f"No resources found for mood {mood_label}, falling back to Neutral resources")
            cursor.execute(
                "SELECT title, link FROM self_help_resources WHERE mood_label = %s",
                ("Neutral 🙂",)
            )
            resources = cursor.fetchall()
            self_help = [{"title": res[0], "link": res[1]} for res in resources]
            
            if not self_help:
                print("No Neutral resources found, returning default message")
                self_help = [{"title": "No resources available at the moment.", "link": "#"}]

        print(f"Returning {len(self_help)} resources for mood {mood_label}")
        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "mood": mood_label,
                "resources": self_help
            }
        })

    except Exception as e:
        print("🔥 ERROR in get_self_help:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# API to Fetch Mood History by User
@app.route('/get_mood_history', methods=['GET'])
@jwt_required()
def get_mood_history():
    print("📅 Get-mood-history endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        query = """
            SELECT id, message, vader_score, bert_label, bert_score, mood_label, timestamp 
            FROM mood 
            WHERE user_id = %s 
            ORDER BY timestamp DESC
        """
        cursor.execute(query, (user_id,))
        mood_history = cursor.fetchall()

        mood_list = [
            {
                "mood_id": mood[0],
                "message": mood[1],
                "vader_score": float(mood[2]) if mood[2] is not None else None,
                "bert_label": mood[3] if mood[3] else None,
                "bert_score": float(mood[4]) if mood[4] is not None else None,
                "mood_label": mood[5],
                "timestamp": str(mood[6])
            }
            for mood in mood_history
        ]

        cursor.close()
        db.close()

        return jsonify({"status": "success", "data": {"mood_history": mood_list}})

    except Exception as e:
        print("🔥 ERROR in get_mood_history:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


# API to Save Quiz Result
@app.route('/save_quiz_result', methods=['POST'])
@jwt_required()
def save_quiz_result():
    print("📝 Save-quiz-result endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        data = request.json
        quiz_title = data.get("quizTitle", "").strip()
        score = data.get("score")
        total_questions = data.get("totalQuestions")
        category = data.get("category", "").strip()
        result_level = data.get("resultLevel", "").strip()
        elapsed_time = data.get("elapsedTime")  # New field for elapsed time in seconds

        # Validate required fields
        if not all([quiz_title, score is not None, total_questions is not None, category, result_level]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        if not isinstance(score, (int, float)) or not isinstance(total_questions, int):
            return jsonify({"status": "error", "message": "Score must be a number and total questions must be an integer"}), 400

        if score < 0 or total_questions <= 0 or score > total_questions:
            return jsonify({"status": "error", "message": "Invalid score or total questions"}), 400

        if elapsed_time is None or not isinstance(elapsed_time, (int, float)) or elapsed_time < 0:
            return jsonify({"status": "error", "message": "Elapsed time must be a non-negative number"}), 400

        # Insert quiz result into the database
        query = """
            INSERT INTO quiz_results (user_id, quiz_title, score, total_questions, category, result_level, elapsed_time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        cursor.execute(query, (user_id, quiz_title, score, total_questions, category, result_level, elapsed_time))
        db.commit()

        quiz_result_id = cursor.lastrowid
        cursor.close()
        db.close()

        print(f"✅ Quiz result saved for user_id: {user_id}, quiz_result_id: {quiz_result_id}")
        return jsonify({
            "status": "success",
            "data": {
                "quiz_result_id": quiz_result_id,
                "message": "Quiz result saved successfully"
            }
        })

    except Exception as e:
        print("🔥 ERROR in save_quiz_result:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# API to Get Quiz History for a User
@app.route('/get_quiz_history', methods=['GET'])
@jwt_required()
def get_quiz_history():
    print("📜 Get-quiz-history endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        query = """
            SELECT id, quiz_title, score, total_questions, category, result_level, elapsed_time, timestamp
            FROM quiz_results
            WHERE user_id = %s
            ORDER BY timestamp DESC
        """
        cursor.execute(query, (user_id,))
        quiz_results = cursor.fetchall()

        quiz_history = [
            {
                "id": result["id"],
                "quiz_title": result["quiz_title"],
                "score": result["score"],
                "total_questions": result["total_questions"],
                "category": result["category"],
                "result_level": result["result_level"],
                "elapsed_time": result["elapsed_time"],  # Include elapsed time
                "timestamp": str(result["timestamp"])
            }
            for result in quiz_results
        ]

        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "quiz_history": quiz_history,
                "message": f"Retrieved {len(quiz_history)} quiz results for user {user_id}"
            }
        })

    except Exception as e:
        print("🔥 ERROR in get_quiz_history:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500



# API to Get Mini-Game Suggestions Based on Mood
@app.route('/get_mini_games', methods=['GET'])
@jwt_required()
def get_mini_games():
    print("🎮 Get-mini-games endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        # Fetch the user's latest mood
        cursor.execute(
            "SELECT mood_label FROM mood WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1",
            (user_id,)
        )
        mood_result = cursor.fetchone()
        mood_label = mood_result["mood_label"] if mood_result else "Neutral 🙂"
        print(f"Latest mood for user {user_id}: '{mood_label}' (HEX: {mood_label.encode('utf-8').hex()})")

        # Validate the mood_label
        allowed_moods = ["Happy 😊", "Sad 😔", "Angry 😡", "Stressed 😟", "Tired 😴", "Excited 🎉", "Neutral 🙂"]
        if mood_label not in allowed_moods:
            print(f"Invalid mood_label detected: '{mood_label}', defaulting to Neutral")
            mood_label = "Neutral 🙂"

        # Normalize mood_label by stripping the emoji for comparison
        # Extract the base mood (e.g., "Tired" from "Tired 😴")
        mood_base = mood_label.split()[0]  # Get the first word (e.g., "Tired")
        print(f"Normalized mood base: {mood_base}")

        # Fetch mini-games based on the normalized mood_label
        query = """
            SELECT id, title, description, stress_level, mood_label, game_type 
            FROM mini_games 
            WHERE SUBSTRING_INDEX(mood_label, ' ', 1) = %s 
            LIMIT 3
        """
        cursor.execute(query, (mood_base,))
        games = cursor.fetchall()
        print(f"Games found for mood '{mood_label}' (base: {mood_base}): {games}")

        if not games:
            # Fallback to Neutral mood games if none match the user's mood
            print(f"No games found for mood '{mood_label}', falling back to Neutral games")
            cursor.execute(
                "SELECT id, title, description, stress_level, mood_label, game_type FROM mini_games WHERE SUBSTRING_INDEX(mood_label, ' ', 1) = 'Neutral' LIMIT 3"
            )
            games = cursor.fetchall()
            print(f"Fallback games for Neutral 🙂: {games}")

        if not games:
            print("No Neutral games found, returning empty list")
            games = []

        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "data": {
                "games": games,
                "mood_label": mood_label
            }
        }), 200

    except Exception as e:
        print("🔥 ERROR in get_mini_games:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500
        
# API to Log Game Interaction
@app.route('/log_game_interaction', methods=['POST'])
@jwt_required()
def log_game_interaction():
    print("📊 Log-game-interaction endpoint called")
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id)
        db = get_db_connection()
        cursor = db.cursor()

        data = request.json
        game_id = data.get("game_id")

        if not game_id:
            return jsonify({"status": "error", "message": "Game ID is required"}), 400

        # Check if the game exists
        cursor.execute("SELECT id FROM mini_games WHERE id = %s", (game_id,))
        if not cursor.fetchone():
            return jsonify({"status": "error", "message": "Game not found"}), 404

        # Log the interaction
        cursor.execute(
            "INSERT INTO user_game_interactions (user_id, game_id) VALUES (%s, %s)",
            (user_id, game_id)
        )
        db.commit()

        cursor.close()
        db.close()

        return jsonify({
            "status": "success",
            "message": "Game interaction logged successfully"
        }), 200

    except Exception as e:
        print("🔥 ERROR in log_game_interaction:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

def is_admin_user(user_id):
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id FROM admins WHERE id = %s", (user_id,))
        admin = cursor.fetchone()
        cursor.close()
        db.close()
        return admin is not None
    except Exception as e:
        print(f"🔥 Error checking admin status: {str(e)}")
        return False

# Admin Login Route
@app.route('/admin/login', methods=['POST'])
def admin_login():
    print("🔐 Admin login endpoint called")
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()

        if not email or not password:
            return jsonify({"status": "error", "message": "Email and password are required"}), 400

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, password FROM admins WHERE email = %s", (email,))
        admin = cursor.fetchone()
        cursor.close()
        db.close()

        if not admin:
            print("Admin not found")
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401

        admin_id, stored_password = admin['id'], admin['password']
        # Compare plaintext passwords directly
        if password != stored_password:
            print("Password mismatch")
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401

        access_token = create_access_token(identity=str(admin_id), expires_delta=timedelta(days=1))
        print(f"✅ Admin login successful for admin_id: {admin_id}")
        return jsonify({
            "status": "success",
            "data": {
                "access_token": access_token,
                "admin_id": admin_id,
                "message": "Admin login successful"
            }
        })

    except Exception as e:
        print(f"🔥 Error in admin_login: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin Route to List All Users
@app.route('/admin/users', methods=['GET'])
@jwt_required()
def admin_get_users():
    print("📋 Admin get-users endpoint called")
    try:
        admin_id = get_jwt_identity()
        if not is_admin_user(admin_id):
            return jsonify({"status": "error", "message": "Unauthorized: Admin access required"}), 403

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, full_name, email, created_at FROM user")
        users = cursor.fetchall()
        cursor.close()
        db.close()

        user_list = [
            {
                "id": user["id"],
                "full_name": user["full_name"],
                "email": user["email"],
                "created_at": str(user["created_at"])
            }
            for user in users
        ]

        return jsonify({
            "status": "success",
            "data": {
                "users": user_list
            }
        })

    except Exception as e:
        print(f"🔥 Error in admin_get_users: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin Route to Add a New User
@app.route('/admin/users', methods=['POST'])
@jwt_required()
def admin_add_user():
    print("➕ Admin add-user endpoint called")
    try:
        admin_id = get_jwt_identity()
        if not is_admin_user(admin_id):
            return jsonify({"status": "error", "message": "Unauthorized: Admin access required"}), 403

        data = request.get_json()
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()

        if not full_name or not email or not password:
            return jsonify({"status": "error", "message": "Full name, email, and password are required"}), 400

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        # Check if email already exists
        cursor.execute("SELECT id FROM user WHERE email = %s", (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            db.close()
            return jsonify({"status": "error", "message": "Email already exists"}), 400

        # Insert new user
        cursor.execute(
            "INSERT INTO user (full_name, email, password, date_of_birth, gender, phone_number) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (full_name, email, password, '1990-01-01', 'Other', '1234567890')  # Default values for required fields
        )
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"status": "success", "message": "User added successfully"}), 201

    except Exception as e:
        print(f"🔥 Error in admin_add_user: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin Route to Delete a User
@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def admin_delete_user(user_id):
    print(f"🗑️ Admin delete-user endpoint called for user_id: {user_id}")
    try:
        admin_id = get_jwt_identity()
        if not is_admin_user(admin_id):
            return jsonify({"status": "error", "message": "Unauthorized: Admin access required"}), 403

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        # Check if user exists
        cursor.execute("SELECT id FROM user WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            cursor.close()
            db.close()
            return jsonify({"status": "error", "message": "User not found"}), 404

        # Delete related records (cascade should handle this if set in DB, but let's be explicit)
        cursor.execute("DELETE FROM chat WHERE user_id = %s", (user_id,))
        cursor.execute("DELETE FROM user_activity_logs WHERE user_id = %s", (user_id,))
        cursor.execute("DELETE FROM mood_trends WHERE user_id = %s", (user_id,))
        cursor.execute("DELETE FROM user WHERE id = %s", (user_id,))
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"status": "success", "message": "User deleted successfully"}), 200

    except Exception as e:
        print(f"🔥 Error in admin_delete_user: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin Route to View a User's Chats
@app.route('/admin/chats/<int:user_id>', methods=['GET'])
@jwt_required()
def admin_get_chats(user_id):
    print(f"💬 Admin get-chats endpoint called for user_id: {user_id}")
    try:
        admin_id = get_jwt_identity()
        if not is_admin_user(admin_id):
            return jsonify({"status": "error", "message": "Unauthorized: Admin access required"}), 403

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, session_id, message, response, timestamp FROM chat WHERE user_id = %s ORDER BY timestamp DESC", (user_id,))
        chats = cursor.fetchall()
        cursor.close()
        db.close()

        chat_list = [
            {
                "chat_id": chat["id"],
                "session_id": chat["session_id"],
                "message": chat["message"],
                "response": chat["response"],
                "timestamp": str(chat["timestamp"])
            }
            for chat in chats
        ]

        return jsonify({
            "status": "success",
            "data": {
                "chats": chat_list
            }
        })

    except Exception as e:
        print(f"🔥 Error in admin_get_chats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin Route to View User Activity Analytics
@app.route('/admin/analytics/activity/<int:user_id>', methods=['GET'])
@jwt_required()
def admin_get_user_activity(user_id):
    print(f"📊 Admin get-user-activity endpoint called for user_id: {user_id}")
    try:
        admin_id = get_jwt_identity()
        if not is_admin_user(admin_id):
            return jsonify({"status": "error", "message": "Unauthorized: Admin access required"}), 403

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM user_activity_logs WHERE user_id = %s ORDER BY timestamp DESC LIMIT 10",
            (user_id,)
        )
        activities = cursor.fetchall()
        cursor.close()
        db.close()

        activity_list = [
            {
                "id": activity["id"],
                "user_id": activity["user_id"],
                "activity_type": activity["activity_type"],
                "description": activity["description"],
                "timestamp": str(activity["timestamp"])
            }
            for activity in activities
        ]

        return jsonify({
            "status": "success",
            "data": {
                "activities": activity_list
            }
        })

    except Exception as e:
        print(f"🔥 Error in admin_get_user_activity: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin Route to View User Mood Trends
@app.route('/admin/analytics/mood-trends/<int:user_id>', methods=['GET'])
@jwt_required()
def admin_get_mood_trends(user_id):
    print(f"📈 Admin get-mood-trends endpoint called for user_id: {user_id}")
    try:
        admin_id = get_jwt_identity()
        if not is_admin_user(admin_id):
            return jsonify({"status": "error", "message": "Unauthorized: Admin access required"}), 403

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT mood_label, date, occurrence_count FROM mood_trends WHERE user_id = %s ORDER BY date DESC LIMIT 7",
            (user_id,)
        )
        trends = cursor.fetchall()
        cursor.close()
        db.close()

        trend_list = [
            {
                "mood_label": trend["mood_label"],
                "date": str(trend["date"]),
                "occurrence_count": trend["occurrence_count"]
            }
            for trend in trends
        ]

        return jsonify({
            "status": "success",
            "data": {
                "mood_trends": trend_list
            }
        })

    except Exception as e:
        print(f"🔥 Error in admin_get_mood_trends: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)