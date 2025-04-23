import streamlit as st

# Light mode
def apply_light_styles():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f9f9f9;
        }

        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }

        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 6px;
            height: 3em;
            font-weight: 600;
        }

        .stDownloadButton > button {
            background-color: #3498db;
            color: white;
            border-radius: 6px;
            height: 3em;
            font-weight: 600;
        }

        .streamlit-expanderHeader {
            font-size: 1.1rem;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# Dark Mode
def apply_dark_styles():
    st.markdown(
        """
        <style>
        .main {
            background-color: #111111;
        }

        h1, h2, h3 {
            color: #f1f1f1;
            font-family: 'Segoe UI', sans-serif;
        }

        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border-radius: 6px;
            height: 3em;
            font-weight: 600;
        }

        .stDownloadButton > button {
            background-color: #d62728;
            color: white;
            border-radius: 6px;
            height: 3em;
            font-weight: 600;
        }

        .streamlit-expanderHeader {
            font-size: 1.1rem;
            color: #eee;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
