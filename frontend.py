import streamlit as st
from backend import analyze_sentiment,bulk_analyze, authenticate_user, analyze, register_user,get_user_role, save_feedback, get_all_users,update_user_role,delete_user,get_all_feedback,delete_reports_between

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from datetime import datetime, time
from pytz import timezone, utc
from db_config import engine, get_db
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from io import StringIO
import pdfkit
import tempfile
import plotly.io as pio
import base64
from io import BytesIO


import datetime
Session = sessionmaker(bind=engine)
session = Session()

st.set_page_config(page_title = "Sentiment analysis on X reviews", page_icon = ":smile:", layout = "wide")
# Session State Initialization
if "show_homepage" not in st.session_state:
    st.session_state["show_homepage"] = True  
if "loggedIn" not in st.session_state:
    st.session_state["loggedIn"] = False
if "register" not in st.session_state:
    st.session_state["register"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "show_rules" not in st.session_state:
    st.session_state["show_rules"] = False
if "show_login" not in st.session_state:
    st.session_state["show_login"] = False

# Define UI sections
headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logoutSection = st.container()
registrationSection = st.container()


      

def LoggedOut_clicked():
    st.session_state['loggedIn'] = False
    st.session_state['role'] = None
    st.session_state['username'] = None
    st.session_state['user_id'] = None
    st.session_state['show_homepage'] = True  # Reset to landing page


def LoggedIn_Clicked(user_name, password):
    user_id, role = authenticate_user(user_name, password)  # Validate login
    if role:
        st.session_state['loggedIn'] = True
        st.session_state['username'] = user_name
        st.session_state['user_id'] = int(user_id)
        st.session_state['role'] = role
        
         # Display user profile with icon
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("C:\\Users\\dell\\Pictures\\ICON USER.PNG", width=60)  # Ensure the image file exists
        with col2:
            st.markdown("""
                <div style="display: flex; align-items: center;">
                <span style="background-color: #0a0202; padding: 8px 12px; border-radius: 5px; font-weight: bold; font-size: 16px;">
                Welcome, <b>{user_name}</b> ({role})
                </span>
                </div>
                """.format(user_name=user_name, role=role), unsafe_allow_html=True)
            
    else:
        st.session_state['loggedIn'] = False
        st.error("Invalid username or password")

def show_homepage():
    st.markdown(
        """
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Background and body with gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #00A3E0, #A8DADC);  /* Change gradient here */
            background-size: cover;
            background-attachment: fixed;
            animation: fadeIn 1s ease-in;
            color: white;
        }

        /* Transparent header */
        [data-testid="stHeader"] {
            background-color: rgba(0, 0, 0, 0);
        }

        /* Make ALL text white */
        h1, h2, h3, h4, p, li, div, span, label {
            color: white !important;
        }

        /* Heading fonts */
        .big-font {
            font-size: 48px !important;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-font {
            font-size: 24px !important;
            text-align: center;
            margin-bottom: 30px;
            color: #e0e0e0 !important;
        }

        /* Section boxes */
        .column-box {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s, box-shadow 0.3s;
            animation: fadeIn 1.5s ease-in;
        }
        .column-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(255, 255, 255, 0.2);
        }

        /* Beautiful button */
        div.stButton > button {
            background-color: #00BFFF;
            color: white;
            font-size: 20px;
            padding: 12px 30px;
            border: none;
            border-radius: 12px;
            font-weight: bold;
            margin-top: 40px;
            transition: background-color 0.3s;
        }
        div.stButton > button:hover {
            background-color: #009ACD;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Headings ---
    st.markdown("""<div class='big-font'>Welcome to Sentiment Analysis on X Reviews!</div>""", unsafe_allow_html=True)
    st.markdown("""<div class='sub-font'>Understand your users' feelings. Make better decisions today.</div>""", unsafe_allow_html=True)
    st.markdown("---")

    # --- Three sections side-by-side ---
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<div class="column-box">', unsafe_allow_html=True)
        st.markdown("##  Why Use This Platform?")
        st.markdown(
            """
            - 🔍 Analyze single texts or bulk reviews effortlessly.  
            - 📊 Download ready-to-use sentiment reports.  
            - 🧠 Gain quick, actionable insights.  
            - 🔒 Data is processed securely and confidentially.
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="column-box">', unsafe_allow_html=True)
        st.markdown("##  How It Works")
        st.markdown(
            """
            1. **Sign up or log in** to your account.  
            2. **Choose analysis mode**: Single input or CSV bulk upload.  
            3. **Analyze**, view results, and **export your report**.
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="column-box">', unsafe_allow_html=True)
        st.markdown("##  Who Should Use This?")
        st.markdown(
            """
            This platform is ideal for:  
            - Marketing Teams  
            - Customer Support Analysts  
            - Product Managers  
            - Business Owners  
            - Researchers & Students
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Centered Get Started Button ---
    col_center = st.columns([2, 1, 2])
    with col_center[1]:
        if st.button("🚀 Get Started", key="homepage_get_started"):
            st.session_state['show_homepage'] = False
            st.session_state['show_login'] = True


    
def show_login_page():
    page_bg_img = """
    <style>
    /* Gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);  /* Change gradient here */
        background-size: cover;
        background-attachment: fixed;
        animation: fadeIn 1s ease-in;
        color: white;
    }

    [data-testid="stHeader"]{
        background-color: rgba(0,0,0,0);  /* Transparent header */
    }

    /* Style Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.2); /* Transparent White */
        color: black;
        border: 1px solid white;
    }

    /* Style Buttons */
    div.stButton > button {
        background-color: #4CAF50 !important; /* Green */
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #45a049 !important; /* Slightly darker green */
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    with loginSection:
        st.markdown("<h1 style='text-align: center; color: white;'>Sentiment Analysis on X reviews</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Login and get the chance to see your users' opinions today!</h2>", unsafe_allow_html=True)
        
        st.markdown("""
    <style>
    /* Change all label text to white for better contrast */
    label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
        # Customizing layout
        user_name = st.text_input(" 👤 Username", placeholder="Enter your username", key="login_username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password",key="login_password" )

        
        st.button("Login", on_click=LoggedIn_Clicked, args=(user_name, password))
        st.markdown("<h4 style='text-align: center; color: white;'>Don't have an account? Register with us today!</h4", unsafe_allow_html=True)
        # Toggle to Registration Page
        if st.button("Go to Register"):
            st.session_state['register'] = True
        if st.button("Back to Homepage"):
            st.session_state['show_homepage'] = True
            st.session_state['show_login'] = False
        st.markdown("<br>", unsafe_allow_html=True)

if "users" not in st.session_state:
    st.session_state.users = {}

def register_user(username, email, password):
    users = st.session_state.users
    if username in users:
        return False  # Username already exists
    users[username] = {"email": email, "password": password}
    return True  # Successfully registered


def show_registration_page():
    page_bg_img = """
    <style>
    /* Gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);  /* Change gradient here */
        background-size: cover;
        background-attachment: fixed;
        animation: fadeIn 1s ease-in;
        color: white;
    }

    [data-testid="stHeader"]{
        background-color: rgba(0,0,0,0);  /* Transparent header */
    }

    /* Style Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.2); /* Transparent White */
        color: black;
        border: 1px solid white;
    }

    /* Style Buttons */
    div.stButton > button {
        background-color: green !important;  /* Green */
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #45a049 !important;  /* Slightly darker green */
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    with registrationSection:
        st.markdown("<h1 style='text-align: center; color: white;'>Sentiment analysis on X reviews</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: white;'>User Registration</h1>", unsafe_allow_html=True)
        
        users = st.session_state.users
        
        def is_valid_password(new_password):
            return (len(new_password) >= 8 and
            any(c.isupper() for c in new_password) and
            any(c.isdigit() for c in new_password) and
            any(c in "@#$%^&*()!?" for c in new_password))
        st.markdown("""
            <div style="background-color: rgba(0, 0, 0, 0.6); padding: 10px; border-radius: 10px; color: white;">
                <h4>Password Requirements</h4>
                <ul>
                    <li>At least 8 characters</li>
                    <li>At least one <strong>uppercase</strong> letter</li>
                    <li>At least one <strong>number</strong></li>
                    <li>At least one <strong>special character</strong> (@, #, $, etc.)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
    <style>
    /* Change all label text to white for better contrast */
    label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
        new_username = st.text_input("New Username", placeholder="Choose a username", key="register_username_unique")
        email = st.text_input("Email address", placeholder="Enter your email", key="register_email")
        new_password = st.text_input("New Password", type="password", placeholder="Create a password",key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password",placeholder="Confirm password", key = "confirm_password")
        
        
    if st.button("Register"):
        if not new_username or not email or not new_password or not confirm_password:
            st.warning("All fields are required!")
        elif "@" not in email:
            st.warning("Enter a valid email address!")
        elif new_password != confirm_password:
            st.warning("Passwords do not match!")
        elif not is_valid_password(new_password):
            st.warning("Password must meet all requirements!")
        else:
            success = register_user(new_username, email, new_password)
            if success:
                st.success("Registration successful! You can now log in.")
            else:
                st.warning(f"The username '{new_username}' already exists. Please choose another one.")
    
    if st.button("Go to Login"):
        st.session_state['register'] = False
    if st.button("Back to Homepage"):
            st.session_state['show_homepage'] = True
            st.session_state['register'] = False



def show_logout_page():
    with logoutSection:
        st.button("Logout", key="logout", on_click=LoggedOut_clicked)

def user_management():
    """Admin dashboard for managing users"""
    st.title("User Management")

    users = get_all_users()  # Fetch users from backend

    if not users:
        st.warning("No users found.")
        return

    # Display users in a table
    st.subheader("All Users")
    for user in users:
        st.write(f"👤 **{user.username}** | 📧 {user.email} | 🛠 Role: {user.role}")

        # Update user role
        new_role = st.selectbox(f"Change role for {user.username}", ["user", "admin", "analyst"], index=["user", "admin", "analyst"].index(user.role))
        if st.button(f"Update Role {user.username}"):
            response = update_user_role(user.user_id, new_role)
            st.success(response["message"])

        # Delete user
        if st.button(f"Delete {user.username}", key=f"delete_{user.user_id}"):
            response = delete_user(user.user_id)
            st.warning(response["message"])
def view_feedback():
    """Admin view for feedback."""
    st.markdown("# 📝 User Feedback")

    # Fetch feedback from backend
    feedbacks = get_all_feedback()

    if not feedbacks:
        st.info("No feedback available.")
        return

    # Convert data into a DataFrame
    df = pd.DataFrame([(f.user_id, f.feedback_text, f.feedback_date) for f in feedbacks], 
                      columns=["User ID", "Feedback", "Date Submitted"])
     # Display the feedback in a table
    st.dataframe(df)
def show_main_page():
    with mainSection:
        # Add gradient background to sidebar and main content
        st.markdown(
    """
    <style>
    /* Gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        color: white;
    }

    /* Style the image caption "Sentiment Analysis" */
    [data-testid="stSidebar"] img + div {
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
        margin-top: 5px;
    }

    /* Style the sidebar header text like "Choose your filter:" */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6) !important;
    }
     /* Header Styling */
    [data-testid="stHeader"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC); /* Gradient header */
        color: black !important; /* Ensures the header text is black */
    }
    /* Style text inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid white;
    }

    /* Style all buttons */
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #45a049 !important;
    }

    /* Style only Logout button */
    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-size: 14px !important;
        padding: 6px 12px !important;
        width: auto !important;
        border-radius: 6px;
        font-weight: bold;
    }

    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button:hover {
        background-color: #e60000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



        # Sidebar content
        st.sidebar.image("APP/images/sentiment_logo-removebg-preview.png", caption="Sentiment Analysis")
        st.sidebar.header("Choose your filter:")
        
        if st.session_state["role"] == "admin":
            menu = st.sidebar.radio("Options", ["Single Input", "Bulk Analysis", "Reports", "Feedback", "User Management", "Help Center"], key="sidebar_menu")
        elif st.session_state["role"] == "user":
            menu = st.sidebar.radio("Options", ["Single Input", "Bulk Analysis", "Help Center"], key="sidebar_menu")
        elif st.session_state["role"] == "analyst":
            menu = st.sidebar.radio("Options", ["Single Input", "Bulk Analysis", "Reports", "Help Center"], key="sidebar_menu")
        else:
            st.warning("Unauthorized access!")

        # Main page logic for menu options
        if menu == "Single Input":
            sentiment_analysis()
        elif menu == "Bulk Analysis":
            bulk_analysis()
        elif menu == "Reports":
            reports()
        elif menu == "Help Center":
            show_help_center()
        elif menu == "User Management" and st.session_state["role"] == "admin":
            user_management()
        elif menu == "Feedback" and st.session_state.get("role") == "admin":
            view_feedback()

        if st.session_state.get("role") != "admin":  # Hide from admin
            if "user_feedback" not in st.session_state:
                st.session_state["user_feedback"] = ""
            if "clear_feedback" not in st.session_state:
                st.session_state["clear_feedback"] = False

            # Reset the feedback if triggered
            if st.session_state.clear_feedback:
                st.session_state.user_feedback = ""
                st.session_state.clear_feedback = False
            feedback = st.sidebar.text_area("What do you think about this app?", key="user_feedback")
            if st.sidebar.button("Submit Feedback"):
                if feedback.strip():
                    response = save_feedback(st.session_state["user_id"], feedback)
                    if "message" in response:
                        st.sidebar.success("✅ Thank you for your feedback!")
                        st.session_state.clear_feedback = True
                    else:
                        st.sidebar.error(f"⚠️ Error saving feedback: {response['error']}")
                else:
                    st.sidebar.warning("⚠️ Feedback cannot be empty.")


def sentiment_analysis():
    st.markdown(
    """
    <style>
    /* Gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        color: white;
    }

    /* Style the image caption "Sentiment Analysis" */
    [data-testid="stSidebar"] img + div {
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
        margin-top: 5px;
    }

    /* Style the sidebar header text like "Choose your filter:" */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6) !important;
    }
     /* Header Styling */
    [data-testid="stHeader"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC); /* Gradient header */
        color: black !important; /* Ensures the header text is black */
    }
    /* Style text inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid white;
    }

    /* Style all buttons */
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #45a049 !important;
    }

    /* Style only Logout button */
    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-size: 14px !important;
        padding: 6px 12px !important;
        width: auto !important;
        border-radius: 6px;
        font-weight: bold;
    }

    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button:hover {
        background-color: #e60000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.subheader("Sentiment Analysis Tool")
    text_input = st.text_area("Enter a review:")
    
    if st.button("Analyze Sentiment"):
        if not text_input.strip():
            st.warning("Please enter valid text!")
        else:
            sentiment_label, sentiment_score = analyze_sentiment(text_input)
            st.success(f"Sentiment: {sentiment_label}, Score: {sentiment_score}")
            analyze(st.session_state['user_id'], text_input, sentiment_label, sentiment_score, )
            st.toast("Analysis saved to the database.", icon="✅")

def bulk_analysis():
    st.markdown(
    """
    <style>
    /* Gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        color: white;
    }

    /* Style the image caption "Sentiment Analysis" */
    [data-testid="stSidebar"] img + div {
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
        margin-top: 5px;
    }

    /* Style the sidebar header text like "Choose your filter:" */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6) !important;
    }
     /* Header Styling */
    [data-testid="stHeader"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC); /* Gradient header */
        color: black !important; /* Ensures the header text is black */
    }
    /* Style text inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid white;
    }

    /* Style all buttons */
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #45a049 !important;
    }

    /* Style only Logout button */
    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-size: 14px !important;
        padding: 6px 12px !important;
        width: auto !important;
        border-radius: 6px;
        font-weight: bold;
    }

    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button:hover {
        background-color: #e60000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



    st.markdown("# 📁 Bulk Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"])
    
    if uploaded_file:
        with st.spinner("Analyzing file... Please wait."):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='cp1252')

            st.write("✅ **Uploaded File Preview:**")
            st.write(df.head())

            # Debug: Check column names
            st.write("🔍 **Columns in Uploaded File:**")
            st.write(df.columns)
             # Keywords for good and bad columns
            GOOD_KEYWORDS = ["text", "tweet", "review", "message", "content", "comment"]
            BAD_KEYWORDS = ["username", "user", "name", "date", "time", "created", "id"]
       
            def is_text_column(series):
                return pd.api.types.is_string_dtype(series) and series.dropna().apply(lambda x: isinstance(x, str) and len(x.strip()) > 3).mean() > 0.8

            candidate_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if is_text_column(df[col]) and not any(bad in col_lower for bad in BAD_KEYWORDS):
                    candidate_columns.append(col)

            # Prioritize best matches (columns containing good keywords)
            priority_columns = [col for col in candidate_columns if any(good in col.lower() for good in GOOD_KEYWORDS)]

            final_columns = priority_columns if priority_columns else candidate_columns

            if not final_columns:
                st.error("❌ No valid text columns found. Please upload a file containing tweet/comment texts.")
                return

            # MULTISELECT (returns list)
            selected_col = st.multiselect("📝 Select the text column for analysis:", final_columns)

            if st.button("🔍 Analyze"):
                if not selected_col:
                    st.error("❌ Please select a column before analyzing!")
                else:
                    selected_column_name = selected_col[0]  # get the selected column safely

                    # Map selected column to 'text' column
                    df['text'] = df[selected_column_name]

                    st.success(f"✅ '{selected_column_name}' selected for analysis.")

                    st.write("📌 **Before Sentiment Analysis:**")
                    st.write("Columns:", df.columns)

                    # Check if text column has empty values
                    if df['text'].isnull().all():
                        st.error("❌ **Selected column contains only empty values. Please choose a valid column.**")
                    else:
                        # Proceed with analysis
                        response = bulk_analyze(st.session_state['user_id'], df)

                        if "error" in response:
                            st.error(f"❌ Error: {response['error']}")
                        else:
                            st.success("✅ Sentiment analysis completed successfully!")
                            st.write(df.head())  # show first rows

                            
                            st.write("📌 **After Sentiment Analysis:**")
                            st.write("Columns:", df.columns)

                            if 'sentiment_label' not in df.columns or 'sentiment_score' not in df.columns:
                                st.error("❌ **Error: Sentiment columns were not created!**")
                                return  

                # Display modified DataFrame
                st.write("✅ **Modified DataFrame Preview:**")
                st.write(df.head())
                # 💬 Highlight Most Dominant Comments
                st.subheader("💬 Most Dominant Comments by Sentiment")

                # Ensure required columns exist
                if all(col in df.columns for col in ['sentiment_label', 'sentiment_score', 'text']):
                    # Find strongest comments by sentiment
                    top_positive = df[df['sentiment_label'] == 'Positive'].sort_values(by='sentiment_score', ascending=False).head(1)
                    top_negative = df[df['sentiment_label'] == 'Negative'].sort_values(by='sentiment_score', ascending=True).head(1)
                    top_neutral = df[df['sentiment_label'] == 'Neutral'].sort_values(by='sentiment_score', ascending=False).head(1)

                    # Reusable function
                    def display_comment(label, row, color):
                        if not row.empty:
                            st.markdown(f"### {label} 🟢" if label == "Most Positive" else f"### {label} 🔴" if label == "Most Negative" else f"### {label} 🔵")
                            st.info(f"📝 {row['text'].values[0]}")
                            st.caption(f"Score: {row['sentiment_score'].values[0]}")
                        else:
                            st.write(f"No {label.lower()} comment found.")

                    # Display top comments
                    display_comment("Most Positive", top_positive, "green")
                    display_comment("Most Negative", top_negative, "red")
                    display_comment("Most Neutral", top_neutral, "blue")

                else:
                    st.warning("⚠️ Required columns ('sentiment_label', 'sentiment_score', 'text') not found for dominant comment detection.")


                # Save Analysis to Database (Ensure this function is correct in backend)
                for index, row in df.iterrows():
                    analyze(st.session_state.get('user_id', 'default_user'), row['text'], row['sentiment_label'], row['sentiment_score'])

                st.toast("✅ Bulk Analysis saved to the database.", icon="✅")


                # Ensure created_at exists
                if 'created_at' not in df.columns:
                    df['created_at'] = pd.date_range(end=datetime.datetime.today(), periods=len(df)).strftime('%Y-%m-%d')

                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')


                # Convert sentiment to numeric scores
                df['sentiment_score'] = df['sentiment_label'].map({'Positive': 1, 'Neutral': 2, 'Negative': 0})

                # Calculate 7-day rolling average for trends
                df['rolling_avg'] = df['sentiment_score'].rolling(window=3, min_periods=1).mean()

                # Function to detect declining trends
                def detect_decline(df):
                    last_week = df.tail(7)['rolling_avg']
                    if last_week.mean() < 0.7:
                        return "🚨 Alert: Sentiment is declining! Potential customer dissatisfaction detected."
                    return "✅ Sentiment is stable or improving."

                trend_alert = detect_decline(df)  

                # **Visualization 1: Sentiment Distribution Bar Chart**
                st.subheader("📊 Sentiment Distribution")
                sentiment_counts = df['sentiment_label'].value_counts()
                
                # Create a Seaborn bar chart
                plt.figure(figsize=(8, 5))
                bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=['#db2514', '#9ade0d', '#1E90FF'])#f71111
                
                # Add labels on top of bars
                for bar in bars:
                    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.5, 
                             str(int(bar.get_height())), ha='center', fontsize=12, fontweight='bold')

                plt.title("Sentiment Distribution", fontsize=14, fontweight='bold', color = "white")
                plt.xlabel("Sentiment", fontsize=12,color="white")
                plt.ylabel("Count", fontsize=12, color="white")
                plt.gca().set_facecolor('none')  # Transparent axes
                plt.gcf().patch.set_alpha(0)     # Transparent figure
                st.pyplot(plt)
                plt.close()  # Close plot to prevent overlapping

                # **Visualization 2: Sentiment Distribution Pie Chart**
                st.subheader("📈 Sentiment Distribution Pie Chart")
                # Custom color mapping
                custom_colors = {
                    'positive': 'green',   # Assign 'green' to 'positive' sentiment
                    'negative': 'red',     # Assign 'red' to 'negative' sentiment
                    'neutral': 'blue'      # Assign 'blue' to 'neutral' sentiment
                }
                fig = px.pie(df, names='sentiment_label', title='Sentiment Distribution',  color='sentiment_label',  # Use 'sentiment_label' to color the slices
             color_discrete_map=custom_colors) 
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white")
                ) 
                st.plotly_chart(fig)
            

                # **Show Trend Alert**
                st.subheader("📢 Sentiment Trend Alert")
                st.write(trend_alert)
                st.write(df[['created_at', 'sentiment_label', 'sentiment_score', 'rolling_avg']])


                # **Visualization 3: Sentiment Trend Over Time**
                st.subheader("📉 Sentiment Trend Over Time")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['created_at'], df['rolling_avg'], color='blue', linestyle='-', linewidth=2)
                ax.axhline(0, color='red', linestyle='--', linewidth=1)

                ax.set_title("Sentiment Trend (7-Day Rolling Average)", color='white')
                ax.set_xlabel("Date", color='white')
                ax.set_ylabel("Sentiment Score", color='white')
                ax.set_facecolor('none')  # Transparent
                fig.patch.set_alpha(0)    # Transparent
                ax.tick_params(colors='white')  # White ticks

                st.pyplot(fig)
                plt.close()

def reports():
    st.markdown(
    """
    <style>
    /* Gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        color: white;
    }

    /* Style the image caption "Sentiment Analysis" */
    [data-testid="stSidebar"] img + div {
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
        margin-top: 5px;
    }

    /* Style the sidebar header text like "Choose your filter:" */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6) !important;
    }

     /* Header Styling */
    [data-testid="stHeader"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC); /* Gradient header */
        color: black !important; /* Ensures the header text is black */
    }
    /* Style text inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid white;
    }

    /* Style all buttons */
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #45a049 !important;
    }

    /* Style only Logout button */
    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-size: 14px !important;
        padding: 6px 12px !important;
        width: auto !important;
        border-radius: 6px;
        font-weight: bold;
    }

    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button:hover {
        background-color: #e60000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


    st.markdown("# 📊 Sentiment Reports")
    local_tz = timezone("Africa/Nairobi")

    # Date selection
    today = datetime.datetime.today().date()
    start_date = st.date_input("Start Date", value=today)
    end_date = st.date_input("End Date", value=today)

    # Convert to timezone-aware datetimes
    start_date = local_tz.localize(datetime.datetime.combine(start_date, time.min)).astimezone(utc)
    end_date = local_tz.localize(datetime.datetime.combine(end_date, time.max)).astimezone(utc)

    try:
        user_id = st.session_state.get("user_id")
        role = get_user_role(user_id)

        # Base query for summary
        base_query = """
            SELECT sentiment_label, COUNT(*) as count FROM sentimentresults
            WHERE created_at BETWEEN :start_date AND :end_date
        """
        if role != "admin":
            base_query += " AND user_id = :user_id"
        base_query += " GROUP BY sentiment_label"

        params = {"start_date": start_date, "end_date": end_date}
        if role != "admin":
            params["user_id"] = user_id

        with engine.connect() as conn:
            df_summary = pd.read_sql_query(text(base_query), conn, params=params)

        # Calculate percentages
        total_comments = df_summary['count'].sum()
        df_summary['percentage'] = df_summary['count'] / total_comments * 100

        # Display summary
        st.subheader("📋 Summary Statistics")
        for _, row in df_summary.iterrows():
            st.markdown(f"- **{row['sentiment_label']}**: {row['count']} ({row['percentage']:.2f}%)")

        # Bar chart
        st.subheader("📈 Sentiment Distribution")
        fig = px.bar(df_summary, x='sentiment_label', y='count', color='sentiment_label',
            labels={'count': 'Number of Comments', 'sentiment_label': 'Sentiment'},
            title="Sentiment Breakdown")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig)


        # Comment query
        comment_query = """
            SELECT sentiment_label, text_input, created_at
            FROM sentimentresults
            WHERE created_at BETWEEN :start_date AND :end_date
        """
        if role != "admin":
            comment_query += " AND user_id = :user_id"

        with engine.connect() as conn:
            df_comments = pd.read_sql_query(text(comment_query), conn, params=params)

        st.subheader("💬 Top Representative Comments")
        top_comments = {}
        for label in ['positive', 'negative', 'neutral']:
            top = df_comments[df_comments['sentiment_label'].str.lower() == label].sort_values(by='created_at', ascending=False).head(1)
            if not top.empty:
                top_comments[label] = top.iloc[0]['text_input']
                st.markdown(f"**{label.title()}**: {top_comments[label]}")


        # Find most dominant sentiment
        if not df_summary.empty:
            dominant_sentiment = df_summary.sort_values(by='count', ascending=False).iloc[0]['sentiment_label'].lower()
            dominant_comment = top_comments.get(dominant_sentiment, "No comment available.")


            st.subheader(" Most Dominant Sentiment Summary")
            st.markdown(f"###  **Sentiment:** `{dominant_sentiment.title()}`")
            st.markdown(f" **User said:** \"{dominant_comment}\"")
        else:
            st.warning("⚠️ No sentiment data available to generate a report. Please perform some sentiment analysis first.")
        # Export section
        st.subheader(" Export report")
        try:
    # Generate HTML content
            html_report = f"""
                <h2>Sentiment Summary</h2>
                {df_summary.to_html(index=False)}
                <p><strong>Most Dominant Sentiment:</strong> {dominant_sentiment.title()}</p>
                <h3>Top Comments</h3>
                <ul>
                    {''.join(f"<li><strong>{k.title()}:</strong> {v}</li>" for k, v in top_comments.items())}
                </ul>
            """

        # Chart image as base64
            img_bytes = BytesIO()
            pio.write_image(fig, img_bytes, format='png')
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
            html_report += f'<h3>Chart</h3><img src="data:image/png;base64,{img_b64}" width="600"/>'

            # Save HTML to temp file and convert to PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_html:
                tmp_html.write(html_report.encode('utf-8'))
                tmp_html_path = tmp_html.name

            pdf_path = tmp_html_path.replace('.html', '.pdf')
            pdfkit.from_file(tmp_html_path, pdf_path)
            # Serve the PDF in Streamlit
            with open(pdf_path, "rb") as f:
                b64_pdf = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="sentiment_report.pdf">📄 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)    

        except Exception as pdf_error:
            st.warning(f"PDF generation failed: {pdf_error}")
    except Exception as e:
        st.error(f"Something went wrong while generating the report: {e}")
    if role != "user" and role != "analyst":
    
        st.markdown("### 🗓️ Delete Reports by Date Range")
        start_delete_date = st.date_input("Start Date for Deletion:")
        end_delete_date = st.date_input("End Date for Deletion:")

        if st.button("🗑️ Delete Reports in Selected Range"):
            if start_delete_date and end_delete_date:
                try:
                    delete_reports_between(start_delete_date, end_delete_date)
                    st.success("✅ Reports deleted.")
                    st.rerun()  # Refresh the page
                except Exception as e:
                    st.error(f"Error: {str(e)}")


        
def show_help_center():
    st.markdown(
    """
    <style>
    /* Main Gradient Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }

    /* Sidebar Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC);
        color: white;
    }

    /* Sidebar Logo Caption Styling */
    [data-testid="stSidebar"] img + div {
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
        margin-top: 5px;
    }

    /* Sidebar Header Text Styling */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6) !important;
    }
     /* Header Styling */
    [data-testid="stHeader"] {
        background: linear-gradient(135deg, #00A3E0, #A8DADC); /* Gradient header */
        color: black !important; /* Ensures the header text is black */
    }

    /* Text Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid white;
    }

    /* Normal Buttons */
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #45a049 !important;
    }

    /* Small Logout Button Styling */
    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-size: 14px !important;
        padding: 6px 12px !important;
        width: auto !important;
        border-radius: 6px;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_button"] > button:hover {
        background-color: #e60000 !important;
    }

    /* FAQ Expander Styling */
    div[data-testid="stExpander"] > div:first-child {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: white;
    }

    div[data-testid="stExpanderDetails"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
        color: white !important;
        box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.5); /* Optional: adds soft shadow */
    }

    /* Contact Box Styling */
    .contact-box {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid white;
        color: white;
    }

    /* FAQ Header */
    .faq-header {
        text-align: center;
        font-size: 24px;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    if not st.session_state.get("loggedIn"):
        st.warning("Session expired. Please log in again.")
        return

    # Help Center Content
    st.markdown("# 📚 Help Center")
    st.markdown("<p class='faq-header'>Welcome! Find answers to your questions or contact support.</p>", unsafe_allow_html=True)

    st.markdown("## 📖 Frequently Asked Questions")

    # FAQs
    faqs = {
        "🧠 How do I analyze sentiment?": "Navigate to 'Single Input' or 'Bulk Analysis', enter your text, and click 'Analyze'.",
        "📝 How do I register an account?": "Click the 'Register' button, fill in your details, and submit the form.",
        "📊 What do the sentiment scores mean?": "The scores shows the model's confidence in classifying sentimnents.",
        "📁 What do the different columns after uploading a CSV file mean?": "Columns before analysis are from your file; after analysis, we add sentiment labels and scores.",
        "🔒 Is my data secure?": "Yes! Your data is securely stored and only accessible by authorized users.",
        "📥 Can I download reports?": "Absolutely! Use the 'Export PDF' button to download your reports."
    }

    for question, answer in faqs.items():
        with st.expander(question):
            st.write(answer)

    # Upgrade Account Section
    st.markdown("## 🚀 Upgrade Your Account")
    st.info(
        """Want to become an Analyst to access advanced features like report viewing?
        
        👉 [Send us an email](mailto:support_sentiment@gmail.com) with your **username** and **reason** for requesting an upgrade.
        """,
        icon="💬"
    )

    # Contact Support Section
    st.markdown("## 📞 Contact Support")
    st.markdown(
        """
        <div class="contact-box">
            <p>📧 <b>Email:</b> <a href="mailto:support_sentiment@gmail.com" style="color:white;">support_sentiment@gmail.com</a></p>
            <p>📱 <b>Phone:</b> +254 708 362 963</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if not st.session_state.get('loggedIn', False):
    if st.session_state.get('show_homepage', True):
        show_homepage()
    elif st.session_state.get('register', False):
        show_registration_page()
    else:
        show_login_page()
else:
    show_logout_page()
    show_main_page()

