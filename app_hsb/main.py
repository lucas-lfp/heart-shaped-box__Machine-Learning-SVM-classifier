from flask import Flask, render_template, abort, url_for, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, RadioField
from viz_dicts import viz_data
from utils import load_data, load_raw_data
import joblib
import pandas as pd


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = "22475"

class InputForm(FlaskForm):
    age_input = IntegerField("Age: ")
    weight_input = IntegerField("Weight (kg): ")
    height_input = IntegerField("Height (cm): ")
    aphi_input = IntegerField("Systolic Blood Pressure (mmHg): ")
    aplo_input = IntegerField("Diastolic Blood Pressure (mmHg): ")
    cholesterol_input = RadioField("Cholesterol: ", choices = [(1, "Normal (< 2g/L)"), (2, "Above normal (2 - 2.39 g/L)"), (3, "Well above normal (> 2.40 g/L)")])
    gluc_input = RadioField("Glucose: ", choices = [(1, "Normal (0.70 - 0.99 g/L or 3.9 - 5.5 mmol/L)"), (2, "Above normal (1.0 - 1.25 g/L or 5.6 - 6.9 mmol/L)"), (3, "Well above normal (> 1.26 g/L or > 7.0 mmol/L)")])
 
    submit = SubmitField("Make a Prediction !")

svm = joblib.load('hsb_svm.pkl')
scaler = joblib.load('hsb_scaler.pkl')
le = joblib.load('hsb_le.pkl')

df = load_data()
df_raw = load_raw_data()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/viz-home')
def viz_home():
    return render_template('viz_home.html')

@app.route('/visualization/<viz_id>')
def visualization(viz_id):
    if viz_id not in viz_data:
        abort(404) 

    data = viz_data[viz_id].copy()
    
    data['img'] = url_for('static', filename=data['img'])

    return render_template("viz_base.html", **data)

@app.route('/ml', methods = ['GET', 'POST'])
def ml():
    message = ''
    test_output = ""
    user_input = InputForm()

    if request.method == 'POST':
        message = 'Form submitted sucessfully!'

    if user_input.validate_on_submit():
        user_age = user_input.age_input.data
        user_weight = user_input.weight_input.data
        user_height = user_input.height_input.data
        user_aphi = user_input.aphi_input.data
        user_aplo = user_input.aplo_input.data
        user_cholesterol = int(user_input.cholesterol_input.data)
        user_gluc = int(user_input.gluc_input.data)

        user_height_in_meters = user_height / 100  # Convert height from cm to meters
        user_bmi = round(user_weight / (user_height_in_meters ** 2), 1)

        user_apm = round((user_aphi + 2 * user_aplo) / 3, 1)

        test_data = {
            "age": [user_age],
            "ap_hi": [user_aphi],
            "ap_lo": [user_aplo],
            "cholesterol": [user_cholesterol],
            "gluc": [user_gluc],
            "bmi": [user_bmi],
            "ap_m": [user_apm]
        }

        # Create a DataFrame
        df_subject = pd.DataFrame(test_data)
        df_s1 = df_subject.copy()
        # Preprocess data: label encoding, min-max scaling
        df_subject['cholesterol'] = le.transform(df_subject['cholesterol'])
        df_subject['gluc'] = le.transform(df_subject['gluc'])
        
        cols = df_subject.columns
        df_subject = scaler.transform(df_subject)
        df_subject = pd.DataFrame(df_subject, columns = cols)

        sub_pred = svm.predict(df_subject)
        if sub_pred == [0]:
            test_output = f"""
                        <div >
                        <p style = 'margin-left: 3px; border-left: 5px solid darkgreen; padding-left: 8px; padding-bottom: 5px;'>
                        <b>Based on the information provided, the model predicts that it is unlikely for you to have a cardiovascular disease.</b> 
                        <br>However, it's essential to maintain a healthy lifestyle and consult with a 
                        healthcare professional for personalized advice and preventive measures.
                        </p>
                        <p>
                        <i>The accuracy of this prediction is of 74%</i>.
                        </p>
                        </div>"""
        else:
            test_output = f"""
                        <div class = 'all'>
                        <p style = 'margin-left: 3px; border-left: 5px solid firebrick; padding-left: 8px; padding-bottom: 5px;'>
                        <b>Based on the information provided, the model predicts that you may have a higher risk of cardiovascular disease.</b> 
                        <br>It's important to consult with a healthcare professional for further evaluation and guidance.
                        </p>
                        <p>
                        <i>The accuracy of this prediction is of 74%</i>.
                        </p>
                        </div>
                        """
            
    return render_template("ml.html", user_form = user_input, test_result=test_output)



@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)