import streamlit as st

def about_me():
    st.markdown(f"""
                <div class = 'all'>
                <p>
                <i>Bonjour</i> ! And thank you for the visit. My name is Lucas. I'm a pharmacist,
                with a Ph.D. in Pharmacology, and a data scientist.
                </p>
                <p>
                I've got a strong background in life science, especially in clinical research. I started playing with
                data during my Ph.D., and quickly got hooked. It didn't take me long to realize that
                the tools I used were very... Limited.
                </p>
                <p>
                But the good news is, I kinda was looking for a good reason to get into programming, and ... <i>voilà</i> !
                </p>
                </div>
                <div class = 'all'>
                If you wish to know more about my previous experiences, or other projects
                i worked on, reach me on LinkedIn or GitHub.
                </div>
                <div style = "text-align: center;">
                <a href='https://www.linkedin.com/in/lucasfoulonpinto/'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/2048px-LinkedIn_icon.svg.png' width='40px'></a>
                <a href='https://github.com/lucas-lfp'>
        <img src='https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png' width='50px'></a>
                </div>
                <div class = 'all'>
                You may also reach me at <i>foulongeo@gmail.com</i>
                </div>
                """, unsafe_allow_html= True)
                   
    
    #AJOUTE LINKDN, GIT ETC.
    #Emoji lien vers HSB ?