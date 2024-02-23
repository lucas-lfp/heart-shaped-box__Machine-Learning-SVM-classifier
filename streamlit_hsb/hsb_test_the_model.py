import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sklearn
print(sklearn.__version__)

def test_the_model():
    svm = joblib.load('hsb_svm.pkl')
    scaler = joblib.load('hsb_scaler.pkl')
    le = joblib.load('hsb_le.pkl')


    class Subject:
        def __init__(self, age, weight, height, ap_hi, ap_lo, cholesterol, gluc):
            self.age = age                                 # in years
            self.weight = weight                           # in kg
            self.height = height                           # in cm
            self.ap_hi = ap_hi                             # in mmHg
            self.ap_lo = ap_lo                             # in mmHg
            self.cholesterol = cholesterol                 # categorical: 1 = normal, 2 = above average, 3 = well above average
            self.gluc = gluc                               # categorical: 1 = normal,  2 = above average, 3 = well above average
            self.bmi = self.calculate_bmi()                # calculate BMI
            self.ap_m = self.calculate_ap_m()              # calculate mean arterial pressure
            
        def calculate_bmi(self):
            # Calculate BMI (weight / (height in meters)^2)
            height_m = self.height / 100  # Convert height from cm to meters
            return round(self.weight / (height_m ** 2), 1)
        
        def calculate_ap_m(self):
            # Calculate mean arterial pressure
            return round((self.ap_hi + 2 * self.ap_lo) / 3, 1)
        
        def make_predict(self):
            # Gathers subjects data
            subject_data = {
                "age": [self.age],
                "ap_hi": [self.ap_hi],
                "ap_lo": [self.ap_lo],
                "cholesterol": [self.cholesterol],
                "gluc": [self.gluc],
                "bmi": [self.bmi],
                "ap_m": [self.ap_m]
            }
            
            # Create a DataFrame
            df_subject = pd.DataFrame(subject_data)
            df_s1 = df_subject.copy()
            # Preprocess data: label encoding, min-max scaling
            df_subject['cholesterol'] = le.transform(df_subject['cholesterol'])
            df_subject['gluc'] = le.transform(df_subject['gluc'])
            
            cols = df_subject.columns
            df_subject = scaler.transform(df_subject)
            df_subject = pd.DataFrame(df_subject, columns = cols)
            
            # Make the prediction
            sub_pred = svm.predict(df_subject)
            if sub_pred == [0]:
                st.markdown(f"""
                            <div class = 'all'>
                            <p style = 'margin-left: 3px; border-left: 5px solid darkgreen; padding-left: 8px; padding-bottom: 5px;'>
                            <b>Based on the information provided, the model predicts that it is unlikely for you to have a cardiovascular disease.</b> 
                            <br>However, it's essential to maintain a healthy lifestyle and consult with a 
                            healthcare professional for personalized advice and preventive measures.
                            </p>
                            <p>
                            <i>The accuracy of this prediction is of 74%</i>.
                            </p>
                            </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                            <div class = 'all'>
                            <p style = 'margin-left: 3px; border-left: 5px solid firebrick; padding-left: 8px; padding-bottom: 5px;'>
                            <b>Based on the information provided, the model predicts that you may have a higher risk of cardiovascular disease.</b> 
                            <br>It's important to consult with a healthcare professional for further evaluation and guidance.
                            </p>
                            <p>
                            <i>The accuracy of this prediction is of 74%</i>.
                            </p>
                            </div>
                            """, unsafe_allow_html=True)
        
    st.markdown(f"""
                <div class = 'all'>
                    <h1>Cardiovascular Disease Prediction</h1>
                    <p = 'intro'>
                    Are you at risk of a cardiovascular disease ? This model was trained
                    on data from more than 65000 subjects either with or without cardiovascular
                    disease. Enter your data below to find out. 
                    </p>
                </div>
                
    """, unsafe_allow_html=True)

    age = st.slider("Age (years)", 18, 99, value = 50)
    if age not in range(39,64,1):
        st.markdown(f"""
                    <p>
                    The model was trained on data from subjects aged between 39 and 65 years.
                    Prediction may loose in accuracy for ages outside of this range.
                    </p>
    """, unsafe_allow_html=True)
    weight = st.number_input("Weight (kg)", value = 70)
    height = st.number_input("Height (cm)", value = 170)
    if height <50:
        st.markdown("Please make sure that height is given in cm")
    else: pass
    ap_hi = st.number_input('Systolic Blood Pressure (mmHg)', value = 120)
    if ap_hi <40:
        st.markdown("""Please make sure that blood pressure is given in mmHg.
                    As a reminder, 120/80 (systolic/diastolic) is given in mmHg, and
                    its equivalent in cmHg is 12/8
                    """)
    else: pass
    
    ap_lo = st.number_input('Diastolic Blood Pressure (mmHg)', value = 80)
    if ap_lo <30:
        st.markdown("""Please make sure that blood pressure is given in mmHg.
                    As a reminder, 120/80 (systolic/diastolic) is given in mmHg, and
                    its equivalent in cmHg is 12/8
                    """)
    else: pass

    if ap_lo >= ap_hi:
        st.markdown(f"""There may be an issue with blood pressure values: systolic
                    blood pressure is expected to be higher than diastolic blood pressure.
                    """)

    cholesterol_map = {"Normal (< 2g/L)" : 1, "Above normal (2 - 2.39 g/L)" : 2, "Well above normal (> 2.40 g/L)" : 3}
    cholesterol_str = st.selectbox("Cholesterol", options = list(cholesterol_map.keys()))
    cholesterol = cholesterol_map[cholesterol_str]

    gluc_map = {"Normal (0.70 - 0.99 g/L or 3.9 - 5.5 mmol/L)" : 1, "Above normal (1.0 - 1.25 g/L or 5.6 - 6.9 mmol/L)" : 2, "Well above normal (> 1.26 g/L or > 7.0 mmol/L)" : 3}
    gluc_str = st.selectbox("Glucose", options = list(gluc_map.keys()))
    gluc = gluc_map[gluc_str]

    if st.button(label = "Make a Prediction"):
        user = Subject(age, weight, height, ap_hi, ap_lo, cholesterol, gluc)
        user.make_predict()



