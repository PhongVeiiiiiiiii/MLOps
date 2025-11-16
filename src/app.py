from flask import Flask, render_template, request
import mlflow.sklearn
import numpy as np

app = Flask(__name__)

model_path = "../models/best_model"
model = mlflow.sklearn.load_model(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = request.form.get("features")
        if data:
            try:
                features = np.array([float(x) for x in data.split(",")]).reshape(1, -1)
                pred = model.predict(features)
                prediction = int(pred[0])
            except:
                prediction = "Dữ liệu không hợp lệ"
        else:
            prediction = "Chưa nhập dữ liệu"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
