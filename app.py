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
jwt = JWTManager(app)

# Gemini API Configuration
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# System prompt for mental health support
MENTAL_HEALTH_PROMPT = """
You are MindEase, a compassionate mental health support companion. Your goal is to provide empathetic, supportive, and highly personalized responses to help users with their emotional well-being. Always be kind, understanding, and non-judgmental. Do not diagnose conditions or provide medical advice. Keep responses concise (2-3 sentences) and deeply relevant to the user's message and emotion.
"""

# Custom JWT error handlers for debugging
@jwt.invalid_token_loader
def invalid_token_callback(error):
    print("🔥 JWT Error: Invalid token", str(error))
    return jsonify({"status": "error", "message": "Invalid token", "error": str(error)}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    print("🔥 JWT Error: Token has expired", jwt_payload)
    return jsonify({"status": "error", "message": "Token has expired", "error": "The token has expired"}), 401

@jwt.unauthorized_loader
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
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Audio Processing Functions
def save_audio(audio_samples, temp_file="temp_audio.wav"):
    """Save audio samples to a WAV file using soundfile."""
    try:
        print("Saving audio file...")
        sf.write(temp_file, audio_samples, 16000)  # Save as WAV with 16kHz
        print(f"Audio saved to {temp_file}")
        return temp_file
    except Exception as e:
        print(f"Error in save_audio: {str(e)}")
        return None

def transcribe_audio_to_text(audio_file_path, max_attempts=2):
    """Transcribe audio file to text using SpeechRecognition with retry logic."""
    try:
        if not audio_file_path or not os.path.exists(audio_file_path):
            raise FileNotFoundError("Audio file not found for transcription")
        
        print("Transcribing audio to text...")
        recognizer = sr.Recognizer()
        attempt = 1
        while attempt <= max_attempts:
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = recognizer.record(source, duration=10)  # Increased to 10 seconds
                    text = recognizer.recognize_google(audio_data, language='en-US')
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

# Updated Analyze Endpoint
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
        audio_blob = request.files.get('audio')

        if audio_blob:
            print("Audio input detected")
            data = request.form
            session_id = data.get("session_id")
            conversation_history = json.loads(data.get("conversation_history", "[]"))
        else:
            print("Text input detected")
            data = request.json or {}
            session_id = data.get("session_id")
            user_message = data.get("message", "").strip()
            conversation_history = data.get("conversation_history", [])

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
        bert_result = {'label': 'NEUTRAL', 'score': 0.0}

        if audio_blob:
            # Validate audio blob size
            audio_data = BytesIO(audio_blob.read())
            audio_data.seek(0, os.SEEK_END)
            audio_size = audio_data.tell()
            print(f"Audio blob size: {audio_size} bytes")
            if audio_size < 100:  # Arbitrary minimum size to catch empty/invalid audio
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

            # Read and process audio
            try:
                audio_data.seek(0)  # Reset buffer position
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

            # Save audio to a temporary file for transcription
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

            # Step 1: Transcribe audio to text
            try:
                transcription = transcribe_audio_to_text(temp_file, max_attempts=2)
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

            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                print("Temporary audio file removed")

        # Step 2: Detect emotion using text-based sentiment analysis
        if "[Transcription" not in user_message and "[Voice input" not in user_message:
            try:
                vader_result = analyzer.polarity_scores(user_message)
                bert_result = sentiment_pipeline(user_message)[0]
                vader_compound = vader_result.get("compound", 0.0)
                bert_label = bert_result.get("label", "NEUTRAL")

                if vader_compound <= -0.5 or bert_label == "NEGATIVE":
                    mood_label = "Sad 😔"
                elif vader_compound >= 0.5 or bert_label == "POSITIVE":
                    mood_label = "Happy 😊"
                elif any(keyword in user_message.lower() for keyword in ["tired", "exhausted", "sleepy"]):
                    mood_label = "Tired 😴"
                elif any(keyword in user_message.lower() for keyword in ["angry", "mad", "frustrated"]):
                    mood_label = "Angry 😡"
                elif any(keyword in user_message.lower() for keyword in ["stressed", "anxious", "nervous"]):
                    mood_label = "Stressed 😟"
                elif any(keyword in user_message.lower() for keyword in ["excited", "thrilled", "joyful"]):
                    mood_label = "Excited 🎉"
            except Exception as e:
                print(f"🔥 Error in sentiment analysis: {str(e)}")
                mood_label = "Neutral 🙂"
        else:
            mood_label = "Neutral 🙂"

        print(f"Detected emotion: {mood_label}")

        # Step 3: Generate supportive response with Gemini
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history]) if conversation_history else "No previous conversation."
        sentiment_info = f"Sentiment (VADER): {vader_result.get('compound', 0.0)}, (BERT): {bert_result.get('label', 'NEUTRAL')} (Confidence: {bert_result.get('score', 0.0)})"
        
        prompt = f"""
        {MENTAL_HEALTH_PROMPT}
        Below is the conversation history between the user and the assistant.
        Conversation history:\n{history_text}\n
        Current user message: "{user_message}"\n
        {sentiment_info}\n

        Craft a **highly personalized, empathetic, and supportive** response that:
        - Focuses on the detected emotion (based on the sentiment analysis) and validates the user's feelings in a natural, meaningful way.
        - **Do not repeat the user's exact message or specific details they shared** (e.g., if they said "I failed a test," do not mention "failed a test" in the response).
        - Provides comfort, encouragement, or celebration tailored to the emotion (e.g., for Sad, offer deep empathy; for Happy, celebrate their joy; for Angry, validate and suggest calming steps).
        - If the emotion indicates distress (e.g., Sad, Angry, Stressed), gently suggest a simple, actionable self-help strategy to manage their feelings (e.g., taking a few deep breaths, writing down thoughts, or focusing on a comforting activity).
        - Avoid generic phrases like 'I'm here for you' unless they fit naturally; focus on specific, creative language that feels personal.
        - Keep the response concise (2-3 sentences).

        Examples:
        - Failed Transcription: "I’m sorry, I couldn’t understand your voice message this time. Could you try speaking again or typing how you’re feeling, so I can support you better?"
        - Sad: "I can feel how heavy your heart is right now, and I’m so sorry you’re going through this. Maybe taking a moment to breathe deeply or writing down your thoughts could help ease that weight—would you like to try?"
        - Happy: "Your joy is truly shining through, and it’s so wonderful to feel that warmth from you! What’s bringing this brightness into your day?"
        - Angry: "I can sense how much frustration you’re holding, and it’s completely okay to feel this way. How about taking a few slow breaths to help release some of that tension—want to try it together?"
        - Neutral: "You seem to be in a calm space with your thoughts today, which is a gentle place to be. Is there anything on your mind you’d like to explore further?"
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            chat = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
            bot_response = chat.send_message(prompt).text
            print(f"Gemini response: {bot_response}")
        except Exception as gemini_error:
            print(f"🔥 Gemini Error: {str(gemini_error)}")
            bot_response = "I’m sorry, I couldn’t generate a response due to a technical issue with my language model. Could you try again or type your message instead?"
            mood_label = "Neutral 🙂"

        # Step 4: Fetch self-help resources
        try:
            cursor.execute("SELECT title, link FROM self_help_resources WHERE mood_label = %s", (mood_label,))
            resources = cursor.fetchall()
            self_help = [{"title": res[0], "link": res[1]} for res in resources] or [{"title": "No resources available at the moment.", "link": "#"}]
        except Exception as e:
            print(f"🔥 Error fetching self-help resources: {str(e)}")
            self_help = [{"title": "No resources available at the moment.", "link": "#"}]

        # Step 5: Store chat data
        try:
            cursor.execute("""
                INSERT INTO chat (user_id, session_id, message, response, vader_score, bert_label, bert_score, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (user_id, session_id, user_message, bot_response, 
                  vader_result.get('compound', 0.0), 
                  bert_result.get('label', 'NEUTRAL'), 
                  bert_result.get('score', 0.0)))
        except Exception as e:
            print(f"🔥 Error storing chat data: {str(e)}")
            bot_response = "I’m sorry, I couldn’t save our chat due to a database issue. Let’s try again—what’s on your mind?"
            mood_label = "Neutral 🙂"

        # Step 6: Store mood data
        try:
            cursor.execute("""
                INSERT INTO mood (user_id, session_id, message, vader_score, bert_label, bert_score, mood_label, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (user_id, session_id, user_message, 
                  vader_result.get('compound', 0.0), 
                  bert_result.get('label', 'NEUTRAL'), 
                  bert_result.get('score', 0.0), mood_label))
        except Exception as e:
            print(f"🔥 Error storing mood data: {str(e)}")
            # Continue even if mood storage fails, as it's not critical for the response

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
                "mood_tracked": True
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
                "response": "I’m sorry, something went wrong while processing your message. I’m still here for you—let’s try again. What’s on your mind?",
                "mood_label": "Neutral 🙂",
                "self_help": [{"title": "No resources available at the moment.", "link": "#"}],
                "mood_tracked": False
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

        # Check if a specific mood_label is provided in the query parameters
        mood_label = request.args.get('mood_label', None)
        
        if not mood_label:
            # If no mood_label is provided, fetch the latest mood for the user
            cursor.execute(
                "SELECT mood_label FROM mood WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1",
                (user_id,)
            )
            mood_result = cursor.fetchone()
            mood_label = mood_result[0] if mood_result else "Neutral 🙂"
            print(f"Latest mood for user {user_id}: {mood_label}")
        else:
            print(f"Fetching resources for specified mood: {mood_label}")

        # Validate the mood_label against allowed values (optional, but good practice)
        allowed_moods = ["Happy 😊", "Sad 😔", "Angry 😡", "Stressed 😟", "Tired 😴", "Excited 🎉", "Neutral 🙂"]
        if mood_label not in allowed_moods:
            print(f"Invalid mood_label provided: {mood_label}, defaulting to Neutral")
            mood_label = "Neutral 🙂"

        # Fetch self-help resources for the mood
        cursor.execute(
            "SELECT title, link FROM self_help_resources WHERE mood_label = %s",
            (mood_label,)
        )
        resources = cursor.fetchall()
        self_help = [{"title": res[0], "link": res[1]} for res in resources]

        # If no resources are found for the specific mood, fall back to Neutral resources
        if not self_help:
            print(f"No resources found for mood {mood_label}, falling back to Neutral resources")
            cursor.execute(
                "SELECT title, link FROM self_help_resources WHERE mood_label = %s",
                ("Neutral 🙂",)
            )
            resources = cursor.fetchall()
            self_help = [{"title": res[0], "link": res[1]} for res in resources]

        # If still no resources, provide a default message
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

        query = "SELECT id, message, vader_score, bert_label, bert_score, timestamp FROM mood WHERE user_id = %s ORDER BY timestamp DESC"
        cursor.execute(query, (user_id,))
        mood_history = cursor.fetchall()

        mood_list = [
            {
                "mood_id": mood[0],
                "message": mood[1],
                "vader_score": float(mood[2]) if mood[2] is not None else None,
                "bert_label": mood[3] if mood[3] else None,
                "bert_score": float(mood[4]) if mood[4] is not None else None,
                "timestamp": str(mood[5])
            }
            for mood in mood_history
        ]

        cursor.close()
        db.close()

        return jsonify({"status": "success", "data": {"mood_history": mood_list}})

    except Exception as e:
        print("🔥 ERROR in get_mood_history:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)