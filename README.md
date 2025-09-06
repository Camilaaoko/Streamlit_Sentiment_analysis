# 📊 Sentiment Analysis System

This is a web-based sentiment analysis system designed to analyze user opinions and classify them as **Positive**, **Negative**, or **Neutral**. It supports both single text input and bulk analysis from uploaded CSV files. The system is tailored for analysts, admins, and general users with role-based access and reporting features.

## 🚀 Features

- 🔐 User authentication and registration
- ✍️ Single input sentiment analysis
- 📁 Bulk analysis via CSV upload
- 📊 Visual summary of sentiment results (bar/pie charts)
- 📄 Exportable sentiment reports (PDF format)
- 🗂️ Sentiment result history per user
- 🧠 RoBERTa-based machine learning model for accurate sentiment classification
- 🧼 Admin tools for managing users and deleting sentiment data
- 🆘 Built-in Help Center with FAQs and contact support

## 🧰 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python, SQLAlchemy, Pandas
- **Database**: MySQL
- **Model**: RoBERTa (`transformers` library)
- **PDF Export**: pdfkit, HTML templating
- **Deployment**: Localhost

## 🧑‍💼 Roles & Access

| Role    | Capabilities                                                  |
|---------|---------------------------------------------------------------|
| User    | Run sentiment analysis, view history                          |
| Analyst | Access sentiment summaries and download reports               |
| Admin   | Full access – delete reports, manage users, view all feedback |

## 📂 Folder Structure


📁 sentiment_analysis_app/ │ ├── 🗃️ model/ # RoBERTa model & tokenizer ├── 📄 pages/ # Streamlit multi-page setup ├── 📦 database/ # SQLAlchemy ORM models ├── 📑 requirements.txt # Python dependencies ├── 🚀 main.py # App entry point ├── 📝 README.md # Project overview
Trained on: twitter airline sentiment

Format: CSV with columns like text, sentiment
Security
Passwords are hashed for secure authentication.

Role-based access controls sensitive actions.

Admin-only deletion and report controls.
