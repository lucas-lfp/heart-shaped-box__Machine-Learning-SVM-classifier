from flask import Flask, render_template, abort, url_for, request
from viz_dicts import viz_data
from utils import load_data, load_raw_data
import joblib


app = Flask(__name__, template_folder='templates')

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
    input_age = 0
    if request.method == 'POST':
        message = 'Form submitted sucessfully!'
        input_age = int(request.form.get('input_age', 50))
        if input_age < 39:
            message = "The model was trained on data from adults above 39 years old. Prediction may lose in accuracy for younger people."
    return render_template("ml.html", message=message, input_age=input_age)

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)