import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from db_config import engine, get_db
from modelTraining import clean_text, convert_numerical_ratings
from datetime import datetime, timedelta
from pytz import timezone, utc
from transformers import pipeline
from sqlalchemy import delete
import bcrypt
from sqlalchemy import  Column, Integer, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from dotenv import load_dotenv
import glob
import re


#database connection setup

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
load_dotenv()
#Database models

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)
    password = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), nullable=False, default='user')
    created_at = Column(DateTime, default=func.now())

class SentimentResults(Base):
    __tablename__ = "sentimentresults"
    analysis_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    text_input = Column(Text, nullable=False)
    sentiment_label = Column(String(20))
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=func.now())

class Feedback(Base):

    __tablename__ = "feedback"
    feedback_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    feedback_text = Column(Text, nullable=False)
    feedback_date = Column(DateTime, default=func.now())
    

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)
USE_ROBERTA = True

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to APP folder

def get_latest_file(pattern):
    """Find the latest file matching the pattern (e.g., sentiment_model_*.pkl)."""
    files = glob.glob(os.path.join(BASE_DIR, 'APP', pattern))
    print(f"Files found for model: {glob.glob(os.path.join('APP', 'sentiment_model_*.pkl'))}")
    print(f"Files found for vectorizer: {glob.glob(os.path.join('APP', 'vectorizer_*.pkl'))}")

    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")
    return max(files, key=os.path.getmtime)  # Return the newest file

# Prevent Alembic from loading the model
if os.getenv("ALEMBIC_RUNNING") != "true":
    try:
        model_path = get_latest_file("sentiment_model_*.pkl")
        vectorizer_path = get_latest_file("vectorizer_*.pkl")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model or vectorizer missing: {e}")

session = SessionLocal()
try:
    test_query = session.query(User).first()
    print("Database connection successful:", test_query)
except Exception as e:
    print("Database connection failed:", e)
finally:
    session.close()

# Load RoBERTa sentiment pipeline once
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Optional: Map RoBERTa labels to friendly names
roberta_label_map = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}
def authenticate_user(username, password):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            st.session_state["user_id"] = user.user_id
            st.session_state["role"] = user.role
            return user.user_id, user.role
        
        st.error("⚠️ Incorrect username or password! Please try again.", icon="🚨")

        return None, None
    finally:
        session.close()
def check_access(role,required_roles):
    """ Restrict access based on user role """
    if "role" not in st.session_state:
        st.error("⚠️ Access Denied! Please log in first.")
        st.stop()

    user_role = st.session_state["role"]
    
    if user_role not in required_roles:
        st.error("⛔ You do not have permission to access this page.")
        st.stop()
# Function to check if a username or email exists
def is_user_exists(username, email, db: Session):
    return db.query(User).filter((User.username == username) | (User.email == email)).first()

def register_user(username, email, password, role="user"):
    db = next(get_db()) 
    try:
        if is_user_exists(username, email, db):
            return {"error": "Username or email already exists!"}
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, role=role)
        db.add(new_user)
        db.commit()
        return {"message": "User registered successfully"}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()


def get_user_role(user_id):
    session = SessionLocal()
    try:
        user = session.query(User.role).filter(User.user_id == user_id).first()
        return user.role if user else None
    finally:
        session.close()

def analyze_sentiment(text):
    """Process the text and return its sentiment and confidence score."""
    if not isinstance(text, str) or text.strip() == "":
        return "Invalid input", 0.0 
    if re.fullmatch(r'\d+|\W+', text):  # Override classification for numbers/gibberish
        return "Neutral", 0.0  

    if USE_ROBERTA:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        confidence = result['score']
        sentiment = roberta_label_map.get(label, 'Unknown')
        return sentiment, round(confidence, 4)
    
    # Traditional model
    converted_text = convert_numerical_ratings(text)
    cleaned_text = clean_text(converted_text)
    tfidf_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_text)[0]
    probabilities = model.predict_proba(tfidf_text)[0]  
    score = probabilities.max()
    sentiment_map = {0: "negative", 2: "positive", 1: "neutral"}
    sentiment_label = sentiment_map[prediction]
    return sentiment_label, round(score, 4)
def load_data():
    query = "SELECT created_at, sentiment_label FROM sentimentresults"
    df = pd.read_sql(query, engine)

    # Ensure 'created_at' is in datetime format
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])  # Remove invalid dates
    df = df.sort_values(by='created_at')  # Ensure sorted data

    return df

df = load_data()

# Fix Sentiment Mapping
df['sentiment_score'] = df['sentiment_label'].apply(lambda label: analyze_sentiment(label))  # Adjust this based on how sentiment score is generated
df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')

# Fill NaN values with 0 or another strategy (e.g., drop rows)
df['sentiment_score'] = df['sentiment_score'].fillna(0)

# Compute Rolling Average
df['rolling_avg'] = df['sentiment_score'].rolling(window=3, min_periods=1).mean()

# Detect Declining Trend
def detect_decline(df):
    last_week = df.tail(7)['rolling_avg']
    if last_week.mean() < 1:  # A more reasonable threshold for "declining"
        return "🚨 Alert: Sentiment is declining! Potential customer dissatisfaction detected."
    elif last_week.mean() < 1.5:  # Mild decline or neutral
        return "⚠️ Caution: Sentiment is slightly declining."
    return "✅ Sentiment is stable or improving."

trend_alert = detect_decline(df)
print(trend_alert)
def generate_report(start_date, end_date, user_id):
    session = SessionLocal()
    local_tz = timezone("Africa/Nairobi")

    try:
         # Convert to datetime objects
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        # Set start to 00:00:00 and end to 23:59:59
        start_date = local_tz.localize(start_date.replace(hour=0, minute=0, second=0)).astimezone(utc)
        end_date = local_tz.localize(end_date.replace(hour=23, minute=59, second=59)).astimezone(utc)

        # Check if user is admin
        role = get_user_role(user_id)

        query = session.query(
            SentimentResults.sentiment_label, func.count(SentimentResults.sentiment_label).label("count")).filter(SentimentResults.created_at.between(start_date, end_date))
              # If user is not admin, filter results to only their reports
        if role != "admin":
            query = query.filter(SentimentResults.user_id == user_id)
        query = query.group_by(SentimentResults.sentiment_label)
        results = query.all()
        
        return [{"sentiment": row.sentiment_label, "count": row.count} for row in results]    
    finally:
        session.close()
def delete_reports_between(start_date, end_date):
    session = SessionLocal()
    try:
        # Convert to datetime if they are just dates
        start = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.max.time())
        
        session.execute(
            delete(SentimentResults).where(SentimentResults.created_at.between(start, end))
        )
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
def save_feedback(user_id, feedback_text):
    session = SessionLocal()
    try:
        new_feedback = Feedback(user_id=user_id, feedback_text=feedback_text)
        session.add(new_feedback)
        session.flush()
        session.commit()
        return {"message": "Feedback saved successfully"}
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
        return {"error": str(e)}
    finally:
        session.close()
def analyze(user_id, text_input, sentiment_label, sentiment_score):
    sentiment_label, sentiment_score = analyze_sentiment(text_input)
    session = SessionLocal()
    try:
        new_result = SentimentResults(
            user_id=user_id,
            text_input=text_input,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score
        )
        session.add(new_result)
        session.flush()
        session.commit()
        return {"sentiment": sentiment_label, "score": sentiment_score}
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}") 
        return {"error": str(e)}
    finally:
        session.close()
def bulk_analyze(user_id, df):
    session = None
    try:
        session = SessionLocal()
        
        # Check if the 'text' column exists
        if 'text' not in df.columns:
            return {"error": "'text' column not found in the DataFrame"}

        # Apply sentiment analysis
        df[['sentiment_label', 'sentiment_score']] = df['text'].apply(
            lambda x: pd.Series(analyze_sentiment(x))
        )

        results = []
        for index, row in df.iterrows():
            try:
                new_result = SentimentResults(
                    user_id=user_id,
                    text_input=row['text'],
                    sentiment_label=row['sentiment_label'],
                    sentiment_score=row['sentiment_score']
                )
                session.add(new_result)
                session.flush()
                results.append({
                    "text": row['text'],
                    "sentiment": row['sentiment_label'],
                    "score": row['sentiment_score']
                })
            except Exception as e:
                print(f"Error processing row {index}: {e}")

        session.commit()
        return {"message": "Bulk analysis completed", "results": results}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if session:
            session.close()
 
def get_all_users():
    """Retrieve all users (admin only)"""
    session = SessionLocal()
    try:
        return session.query(User.user_id, User.username, User.email, User.role, User.created_at).all()
    finally:
        session.close()

def update_user_role(user_id, new_role):
    """Admin can update a user's role"""
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.user_id == user_id).first()
        if user:
            user.role = new_role
            session.commit()
            return {"message": f"User {user.username} role updated to {new_role}"}
        return {"error": "User not found"}
    finally:
        session.close()

def delete_user(user_id):
    """Admin can delete a user"""
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.user_id == user_id).first()
        if user:
            session.delete(user)
            session.commit()
            return {"message": f"User {user.username} deleted successfully"}
        return {"error": "User not found"}
    finally:
        session.close()
def get_all_feedback():
    """Fetch all feedback from the database."""
    session = SessionLocal()
    try:
        return session.query(Feedback).order_by(Feedback.feedback_date.desc()).all()
    finally:
        session.close()