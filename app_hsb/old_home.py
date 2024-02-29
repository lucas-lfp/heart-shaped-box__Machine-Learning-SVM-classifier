import streamlit as st

def home():
    st.markdown(
            f"""
            <div class = 'titlediv'>
                <p class = 'titlep'>
                    <b>Cardiovascular Diseases in Adult Population</b>
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    st.image('streamlit_hsb/heart_rate_img.png', use_column_width=True)
    st.markdown(
        f"""
        <div class = 'all'>
            <p class = 'titlep_2'>
                Machine Learning - Classification Model
            </p>
            <p class = 'title_sp'>
                <i>Data Science with Real-Life Clinical Data</i>
            </p>
        </div>
        """, unsafe_allow_html=True
    )