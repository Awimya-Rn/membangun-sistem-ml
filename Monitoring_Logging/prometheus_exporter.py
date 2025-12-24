from prometheus_client import start_http_server, Counter, Gauge
from flask import Flask, request, jsonify
import psutil
import time
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

REQUEST_COUNT = Counter("request_count", "Total inference request diterima exporter")
REQUEST_LATENCY_AVG = Gauge("request_latency_avg_seconds", "Rata-rata latency inferensi (detik)")

CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage exporter host (%)")
RAM_USAGE = Gauge("ram_usage_percent", "RAM usage exporter host (%)")
DISK_USAGE = Gauge("disk_usage_percent", "Disk usage root (%)")

SLA_BREACH = Counter("sla_breach_count", "Jumlah request lambat > 1 detik")
ERROR_COUNT = Counter("error_count", "Total error selama inferensi")
PAYLOAD_SIZE = Gauge("payload_size_bytes", "Ukuran payload request terakhir (bytes)")

MODEL_ACCURACY = Gauge("model_accuracy", "Akurasi runtime model")
MODEL_F1 = Gauge("model_f1_score", "F1-score runtime model")
MODEL_PRECISION = Gauge("model_precision", "Precision runtime model")
MODEL_RECALL = Gauge("model_recall", "Recall runtime model")

CORRECT_PRED = Counter("correct_predictions", "Total prediksi benar")
TOTAL_PRED = Counter("total_predictions", "Total prediksi dengan label")

y_true_history = []
y_pred_history = []
total_latency = 0.0
latency_count = 0

app = Flask(__name__)

MLFLOW_MODEL_URL = "http://127.0.0.1:5000/invocations"

MODEL_COLUMNS = [
    "age", "gender", "platform", "daily_screen_time_min", "social_media_time_min", 
    "negative_interactions_count", "positive_interactions_count", "sleep_hours", 
    "physical_activity_min", "anxiety_level", "stress_level", "mood_level"
]

def update_system_metrics():
    """Update CPU, RAM, Disk usage info"""
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage("/").percent)

def normalize_predictions(raw_preds):
    """Memastikan format prediksi adalah integer sederhana"""
    normalized = []
    for p in raw_preds:
        if isinstance(p, list):
            normalized.append(int(p[0]))
        elif isinstance(p, str):
            normalized.append(int(float(p)))
        else:
            normalized.append(int(p))
    return normalized

@app.route("/predict", methods=["POST"])
def predict():
    global total_latency, latency_count

    try:
        input_json = request.get_json()
        PAYLOAD_SIZE.set(len(request.data))
        REQUEST_COUNT.inc()

        start_time = time.time()

        model_response = requests.post(
            MLFLOW_MODEL_URL,
            json={
                "dataframe_split": {
                    "columns": MODEL_COLUMNS,
                    "data": input_json["data"]
                }
            },
            timeout=10
        )

        latency = time.time() - start_time
        total_latency += latency
        latency_count += 1
        REQUEST_LATENCY_AVG.set(total_latency / latency_count)

        if latency > 1:
            SLA_BREACH.inc()

        if model_response.status_code != 200:
            ERROR_COUNT.inc()
            return jsonify({"error": "Model service error", "details": model_response.text}), 500

        raw_predictions = model_response.json() 
        if isinstance(raw_predictions, dict) and "predictions" in raw_predictions:
            raw_predictions = raw_predictions["predictions"]
            
        predictions = normalize_predictions(raw_predictions)

        if "label" in input_json:
            y_true = [int(v) for v in input_json["label"]]
            y_pred = predictions

            for t, p in zip(y_true, y_pred):
                TOTAL_PRED.inc()
                y_true_history.append(t)
                y_pred_history.append(p)
                if t == p:
                    CORRECT_PRED.inc()

            if len(y_true_history) >= 5:
                if len(y_true_history) > 1000:
                    y_true_history.pop(0)
                    y_pred_history.pop(0)

                MODEL_ACCURACY.set(accuracy_score(y_true_history, y_pred_history))
                MODEL_PRECISION.set(precision_score(y_true_history, y_pred_history, average="macro", zero_division=0))
                MODEL_RECALL.set(recall_score(y_true_history, y_pred_history, average="macro", zero_division=0))
                MODEL_F1.set(f1_score(y_true_history, y_pred_history, average="macro", zero_division=0))

        update_system_metrics()

        return jsonify({
            "predictions": predictions,
            "latency": latency,
            "system_info": "Metrics Updated"
        })

    except Exception as e:
        ERROR_COUNT.inc()
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Prometheus Metrics berjalan di port 8000")
    start_http_server(8000)
    
    print("🚀 Exporter API berjalan di port 5001")
    app.run(host="0.0.0.0", port=5001)