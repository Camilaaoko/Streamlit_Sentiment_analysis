
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Khalid..12")
DB_NAME = os.getenv("DB_NAME", "sentiment_analysis")

# Database URL
db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/{DB_NAME}"

# Create database engine
engine = create_engine(db_url, echo=True)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

# Function to get a session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
