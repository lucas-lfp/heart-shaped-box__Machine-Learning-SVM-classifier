import streamlit as st

def about_me():
    st.markdown(f"""
                <div class = 'all'>
                <p>
                Bonjour ! And thank you for the visit. My name is Lucas. I'm a pharmacist,
                with a Ph.D. in Pharmacology, and a data scientist.
                </p>
                <p>
                I've got a strong background in life science, especially in clinical research. I started playing with
                data during my Ph.D., and quickly got hooked. Neither did it took me long to realize that
                the tools I used were very... Limited.
                </p>
                <p>
                But the good news is, I kinda was looking for a good reason to get into coding, and here we are !
                </p>
                </div>
                """, unsafe_allow_html= True)
    
    #AJOUTE LINKDN, GIT ETC.
    #Emoji lien vers HSB ?