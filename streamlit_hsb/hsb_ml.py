import streamlit as st
import pandas as pd
import numpy as np
from hsb_home import home
from hsb_project_presentation import project_presentation
from hsb_dataviz import data_viz
from hsb_test_the_model import test_the_model
from hsb_about_me import about_me

## CSS ##
with open("hsb_ml_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

## NAVIGATION ##
navigation = st.sidebar.radio('NAVIGATION', ["Home", "Project Presentation", "Data Viz", "Test the Model", "About Me"])

st.markdown(f"""            <p style = "font-variant: small-caps;">
            -- Heart Shaped Box Project --
            </p>""", unsafe_allow_html=True)

if navigation == "Home":
    home()


elif navigation == "Project Presentation":
    project_presentation()

elif navigation == "Data Viz":
    data_viz()

elif navigation == "Test the Model":
    test_the_model()

elif navigation == "About Me":
    about_me()

else: 
    home()

st.markdown(
    f"""
    <div class='signature'>
        Dr. Lucas Foulon-Pinto<a href="https://www.linkedin.com/in/lucasfoulonpinto/" target="_blank" style = "text-decoration: none; font-size: 20px">🐊</a>
        <br>Pharm.D, Ph.D
        <br>Winter 2024
    </div>
    """, unsafe_allow_html=True
)