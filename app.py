from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("titanic_model.pkl")
fare_by_class  = joblib.load("fare_by_class.pkl")

@app.route("/", methods=['GET', 'POST'])
def home():
    prediction = None
    probablity = None

    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = int(request.form['age'])
        fam = int(request.form['fam'])
        embarked = request.form['embarked']

        fare = fare_by_class.get(pclass)

        user_df = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "fam": fam,
            "Fare": fare,
            "Embarked": embarked
        }])

        pred = model.predict(user_df)[0]
        prob = model.predict_proba(user_df)[0][1]

        prediction = "Survived" if pred == 1 else "Did not Survived"
        probablity = round(prob * 100, 2)

    return render_template(
        "index.html", prediction = prediction, probability=probablity
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
