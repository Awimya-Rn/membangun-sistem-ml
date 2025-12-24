import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates()
    
    df = df.dropna()

    cat_columns = ['gender', 'mental_state']
    
    df= df.drop(columns=['person_name', 'date', 'platform'])
    
    le = LabelEncoder()
    for col in cat_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            print(f"Encoded {col}")

    df.to_csv(output_path, index=False)
    print(f"Data preprocessing selesai. File disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_data('mental_health_sosmed_dataset.csv', 'mh_sosmed_dataset.csv')