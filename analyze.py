from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import pandas as pd
import numpy as np
import joblib



class AnalyzeTransaction:
    def __init__(self, transaction: dict) -> None:
# Load the model and preprocessing objects
        self.model = joblib.load('random_forest_model.joblib')
        self.location_encoder = joblib.load('location_encoder.joblib')
        self.merchant_encoder = joblib.load('merchant_encoder.joblib')
        self.scaler = joblib.load('amount_scaler.joblib')
        
        self.location_columns = self.location_encoder.get_feature_names_out(['location'])
        self.merchant_columns = self.merchant_encoder.get_feature_names_out(['ref'])
        
        self.new_transaction = transaction
        

    def process_transaction(self):

        new_transaction_df = pd.DataFrame([self.new_transaction])
        new_transaction_df['date'] = pd.to_datetime(new_transaction_df['date'])
        new_transaction_df['hour'] = new_transaction_df['date'].dt.hour
        new_transaction_df['day_of_week'] = new_transaction_df['date'].dt.dayofweek

        new_transaction_df['location'] = new_transaction_df['location'].apply(lambda x: f"{x['city']}_{x['province']}")
        new_location_encoded = self.location_encoder.transform(new_transaction_df[['location']])
        new_merchant_encoded = self.merchant_encoder.transform(new_transaction_df[['ref']])
        new_amount_scaled = self.scaler.transform(new_transaction_df[['amount']])

        new_features = pd.concat([
            pd.DataFrame(new_location_encoded, columns=self.location_columns),
            pd.DataFrame(new_merchant_encoded, columns=self.merchant_columns),
            pd.DataFrame(new_amount_scaled, columns=['amount_scaled']),
            new_transaction_df[['hour', 'day_of_week']]
        ], axis=1)
        
        return new_features

    def predict_fraud(self, new_features):
        # Predict the likelihood of fraud
        fraud_probability = self.model.predict_proba(new_features)[:, 1][0]  # Probability of the transaction being fraudulent
        # Output the result
        print(f"Likelihood of fraud: {fraud_probability * 100:.2f}%")
        return fraud_probability
    

if __name__ == '__main__':
    new_transaction = {
            "amount": 500.00,
            "date": "2024-08-01 12:00:00",
            "ref": "Electronic Store",  # Merchant category
            "location": {"city": "cape town", "province": "gauteng"}
    }
    analysis = AnalyzeTransaction(new_transaction)
    features = analysis.process_transaction()
    analysis.predict_fraud(features)