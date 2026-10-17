"""ThoraxAI web application entrypoint.

Launch (Windows):  .venv\\Scripts\\python -m streamlit run app.py
"""

import streamlit as st

PAGES = [
    st.Page("app_pages/home.py", title="Home", default=True),  # default page is served at "/"
    st.Page("app_pages/analyze.py", title="Analyze X-ray", url_path="analyze"),
    st.Page("app_pages/insights.py", title="Insights", url_path="insights"),
    st.Page("app_pages/about.py", title="About", url_path="about"),
]

st.navigation(PAGES, position="top").run()
