from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open("contact_angle_pred_model.bin", "rb") as f_in:
    model = pickle.load(f_in)


# initialize Flask app
app = Flask(__name__) 

# Home route
@app.route("/")
def home():
    return jsonify({"message": "Contact Angle Prediction API",
                    "usage": "POST /predict with JSON payload containing feature data."
                    })

# Prediction route
@app.route("/predict", methods = ["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to DataFrame (expecting single or mutiple rows)
        if isinstance(data, dict):
            data_df = pd.DataFrame([data])
        elif isinstance(data, list):
            data_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format. Expecting JSON object or list of objects."}), 400

        # Make Predictions
        predictions = model.predict(data_df)
        predictions = np.array(predictions, dtype=float).tolist()

        # Return predictions as JSON
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
