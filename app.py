"""
Root-level app.py wrapper for Streamlit
This file allows running 'streamlit run app.py' from the project root
"""
import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import and run the main app from src/
from app import main

if __name__ == "__main__":
    main()
