import json
import joblib
import pandas as pd

def init():
    global model, label_map
    model = joblib.load("approval_model.pkl")
    label_map = joblib.load("label_map.pkl")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        req_type = data.get("request_type")
        desc = data.get("description")

        if not req_type or not desc:
            return {"error": "Both request_type and description are required"}

        df = pd.DataFrame([{"request_type": req_type, "description": desc}])
        pred_label = int(model.predict(df)[0])
        pred_flow = label_map.get(pred_label, "Unknown")

        return {"approval_flow": pred_flow}
    except Exception as e:
        return {"error": str(e)}
