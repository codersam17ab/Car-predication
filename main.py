import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved preprocessing + model pipeline once when Flask starts.
model = joblib.load("model.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the original values from the HTML form.
        year = int(request.form["Year"])

        # Keep this age calculation the same as the notebook.
        age = datetime.datetime.now().year - year

        car = pd.DataFrame([{
            "Age": age,
            "Present_Price": float(request.form["Present_Price"]),
            "Kms_Driven": int(request.form["Kms_Driven"]),
            "Owner": int(request.form["Owner"]),
            "Fuel_Type": request.form["Fuel_Type"],
            "Seller_Type": request.form["Seller_Type"],
            "Transmission": request.form["Transmission"]
        }])

        # Pipeline handles one-hot encoding and prediction.
        prediction = float(model.predict(car)[0])
        output = round(prediction, 2)

        if output < 0:
            message = "Sorry, you cannot sell this car"
        else:
            message = f"You can sell the car at {output} lakhs"

        return render_template("index.html", pred=message)

    except (KeyError, TypeError, ValueError) as error:
        return render_template(
            "index.html",
            pred=f"Please check the input values: {error}"
        )
    except Exception as error:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            pred="Sorry, something went wrong while predicting."
        )


if __name__ == "__main__":
    app.run(debug=True)
