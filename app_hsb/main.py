from flask import Flask, render_template, abort, url_for
from viz_dicts import viz_data
from utils import load_data, load_raw_data

app = Flask(__name__, template_folder='templates')

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

@app.route('/ml')
def ml():
    return render_template("ml.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)