import requests
import time
import random
import json

URL = "http://127.0.0.1:5001/predict"

print(f"Target URL: {URL}")
print("Mengirim data dummy dengan label (Kunci Jawaban)...")

while True:
    try:
        features = [
            random.randint(15, 70),         
            random.choice([0, 1, 2]),       
            random.choice([0, 1, 2]),       
            random.randint(60, 600),        
            random.randint(30, 400),        
            random.randint(0, 10),          
            random.randint(0, 10),          
            round(random.uniform(3.0, 10.0), 1), 
            random.randint(0, 120),         
            random.randint(1, 10),          
            random.randint(1, 10),          
            random.randint(1, 10)           
        ]

        true_label = random.choice([0, 1, 2])

        payload = {
            "data": [features],   
            "label": [true_label] 
        }
        
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            res_json = response.json()
            pred = res_json['predictions'][0]
            print(f"[OK] Label Asli: {true_label} | Prediksi Model: {pred} | Latency: {res_json['latency']:.4f}s")
        else:
            print(f"[ERROR] {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"[CONN ERROR] {e}")
    
    time.sleep(1)