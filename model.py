import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json



class TransactionModel:
    def __init__(self, file: str) -> None:
        self.file = file
        self.transaction_data = self.open_file()
        self.data = pd.DataFrame(self.transaction_data)
        self.user_data = self.data[self.data['user'] == 'user_1']
        self.preprocess_data()
        self.location_encoder = OneHotEncoder(sparse_output=False)
        self.merchant_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        
    def open_file(self) -> list[dict, dict]:
        with open('transactions.json', 'r') as file:
            transaction_data = json.load(file)
        return transaction_data


    def preprocess_data(self):
        self.user_data['date'] = pd.to_datetime(self.user_data['date'])
        self.user_data['hour'] = self.user_data['date'].dt.hour
        self.user_data['day_of_week'] = self.user_data['date'].dt.dayofweek
        self.user_data['location'] = self.user_data['location'].apply(lambda x: f"{x['city']}_{x['province']}")
        
    
    def process_location(self):     
        location_encoded = self.location_encoder.fit_transform(self.user_data[['location']])
        return location_encoded
    
    
    def process_merchant(self):
        merchant_encoded = self.merchant_encoder.fit_transform(self.user_data[['ref']])
        return merchant_encoded
    
    
    def process_amount(self):     
        amount_scaled = self.scaler.fit_transform(self.user_data[['amount']])
        return amount_scaled
    

    def create_feautures_dataframe(self, location_encoded, merchant_encoded, amount_scaled):
        # Create features DataFrame
        location_columns = self.location_encoder.get_feature_names_out(['location'])
        location_df = pd.DataFrame(location_encoded, columns=location_columns)
        merchant_columns = self.merchant_encoder.get_feature_names_out(['ref'])
        merchant_df = pd.DataFrame(merchant_encoded, columns=merchant_columns)
        amount_df = pd.DataFrame(amount_scaled, columns=['amount_scaled'])
        features = pd.concat([
            location_df,
            merchant_df,
            amount_df,
            self.user_data[['hour', 'day_of_week']]
        ], axis=1)

        features['is_fraud'] = self.user_data['event_label']
        X = features.drop(columns=['is_fraud'])
        y = features['is_fraud']
        return (X, y)

    
    def trainXandYsplit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return (X_train, y_train)
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    
    def save_model(self):
        joblib.dump(self.model, 'transaction_model.joblib')
        joblib.dump(self.location_encoder, 'location_encoder.joblib')
        joblib.dump(self.merchant_encoder, 'merchant_encoder.joblib')
        joblib.dump(self.scaler, 'amount_scaler.joblib')


if __name__ == '__main__':
    transactionModel = TransactionModel('transactions.json')
    location = transactionModel.process_location()
    amount = transactionModel.process_amount()
    merchant = transactionModel.process_merchant()
    X, y = transactionModel.create_feautures_dataframe(location, merchant, amount)
    X_train, y_train = transactionModel.trainXandYsplit(X, y)
    transactionModel.train_model(X_train, y_train)
    transactionModel.save_model()