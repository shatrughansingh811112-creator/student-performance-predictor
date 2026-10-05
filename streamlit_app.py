"""
Streamlit Cloud Entrypoint
Redirects execution to app.py to support default deployment settings.
"""
import os
import runpy

if __name__ == "__main__" or "streamlit" in globals():
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    runpy.run_path(app_path, run_name="__main__")
