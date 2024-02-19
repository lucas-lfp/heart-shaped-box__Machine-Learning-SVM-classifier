import streamlit as st
from hsb_home import home
from hsb_project_presentation import project_presentation
from hsb_dataviz import data_viz



## CSS ##
with open("hsb_ml_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

## NAVIGATION ##
navigation = st.sidebar.radio('NAVIGATION', ["Home", "Project Presentation", "Data Viz", "Test the Model", "Discussion", "About", "Contact"])

st.markdown(f"""            <p style = "font-variant: small-caps;">
            -- Heart Shaped Box Project --
            </p>""", unsafe_allow_html=True)

if navigation == "Home":
    home()


elif navigation == "Project Presentation":
    project_presentation()

elif navigation == "Data Viz":
    data_viz()

else: 
    st.markdown("c'est ça qu'est la véritché")

st.markdown(
    f"""
    <div class='signature'>
        Dr. Lucas Foulon-Pinto<a href="https://www.linkedin.com/in/lucasfoulonpinto/" target="_blank" style = "text-decoration: none; font-size: 20px">🐊</a>
        <br>Pharm.D, Ph.D
        <br>Winter 2024
    </div>
    """, unsafe_allow_html=True
)