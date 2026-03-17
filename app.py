import pickle
import numpy as np
import json
from flask import Flask, render_template, request

app = Flask(__name__)

# ================= LOAD MODELS =================
with open("KNNmodel.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("NBmodel.pkl", "rb") as f:
    nb_model = pickle.load(f)

# ================= LOAD JSON FILES =================
with open("KNNtrain.json", "r") as f:
    knn_train = json.load(f)

with open("KNNtest.json", "r") as f:
    knn_test = json.load(f)

with open("NBtrain.json", "r") as f:
    nb_train = json.load(f)

with open("NBtest.json", "r") as f:
    nb_test = json.load(f)

# ================= SPECIES DICTIONARY =================
species_dict = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

# ================= HOME ROUTE =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= PREDICT ROUTE =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])
        algorithm = request.form["algorithm"]

        input_array = np.array([[sl, sw, pl, pw]])

        # -------- KNN --------
        if algorithm == "KNN":
            prediction = knn_model.predict(input_array)[0]
            flower = species_dict[prediction]

            result = {
                "algorithm": "KNN",
                "prediction": flower,
                "train": knn_train,
                "test": knn_test
            }

        # -------- NAIVE BAYES --------
        elif algorithm == "NB":
            prediction = nb_model.predict(input_array)[0]
            flower = species_dict[prediction]

            result = {
                "algorithm": "Naive Bayes",
                "prediction": flower,
                "train": nb_train,
                "test": nb_test
            }

        else:
            result = {"error": "Invalid Algorithm Selected"}

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result={"error": str(e)})


# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=True)
