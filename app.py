import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Distribution Fitter", layout="wide")
st.title("📊 Distribution Fitting Tool")

# Basic structure - we'll build from here
st.write("Welcome to the distribution fitting app!")