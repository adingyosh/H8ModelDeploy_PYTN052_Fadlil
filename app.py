from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("modelprice.pkl", "rb"))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return render_template('predict.html')


@app.route('/predict', methods =['POST'])
def predict():
    data1 = request.form['bedrooms']
    data2 = request.form['bathrooms']
    data3 = request.form['sqft_living']
    data4 = request.form['sqft_lot']
    data5 = request.form['floors']
    data6 = request.form['sqft_above']
    data7 = request.form['sqft_basement']
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7]])
    pred = model.predict(arr)
    return render_template('price.html', data = pred)


if __name__ == '__main__':
    app.run(debug=True)