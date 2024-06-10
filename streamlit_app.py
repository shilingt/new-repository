import streamlit as st
import pandas as pd
import math
from pathlib import Path
import datetime

import random
import numpy as np

import pickle
from joblib import load
from sklearn.preprocessing import LabelEncoder, StandardScaler
      
    
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Exercise Prescription Using Cardiac Risk Level',
    page_icon= 'https://arcadiapittwater.com.au/wp-content/uploads/2022/07/Cardiopulmonary-day-rehabilitation.png', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

# Define the Streamlit app


# Load the trained model
risk_scaler = load('risK_scaler.pkl')