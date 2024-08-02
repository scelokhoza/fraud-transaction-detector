import pandas as pd
import joblib


class AnalyzeTransaction:
    """
    A class to analyze a new transaction, process its features, and predict the probability of fraud.

    Attributes:
        model (RandomForestClassifier): The pre-trained RandomForest model loaded from file.
        location_encoder (OneHotEncoder): Encoder for location data, loaded from file.
        merchant_encoder (OneHotEncoder): Encoder for merchant data, loaded from file.
        scaler (StandardScaler): Scaler for normalizing transaction amounts, loaded from file.
        location_columns (array): Feature names for the location data after encoding.
        merchant_columns (array): Feature names for the merchant data after encoding.
        new_transaction (dict): Dictionary representing the new transaction data.

    Methods:
        process_transaction() -> pd.DataFrame: Processes the new transaction's features to match the training data format.
        predict_fraud(new_features) -> float: Predicts the probability of fraud for the processed transaction.
    """

    def __init__(self, transaction: dict) -> None:
        """
        Initializes the AnalyzeTransaction class with the new transaction data and loads the necessary model and encoders.

        Args:
            transaction (dict): Dictionary containing the new transaction data.
        """
        self.model = joblib.load('transaction_model.joblib')
        self.location_encoder = joblib.load('location_encoder.joblib')
        self.merchant_encoder = joblib.load('merchant_encoder.joblib')
        self.scaler = joblib.load('amount_scaler.joblib')
        
        self.location_columns = self.location_encoder.get_feature_names_out(['location'])
        self.merchant_columns = self.merchant_encoder.get_feature_names_out(['ref'])
        
        self.new_transaction = transaction

    def process_transaction(self) -> pd.DataFrame:
        """
        Processes the new transaction's features to match the format of the training data.

        Returns:
            pd.DataFrame: DataFrame containing the processed features.
        """
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

    def predict_fraud(self, new_features) -> float:
        """
        Predicts the probability of fraud for the processed transaction.

        Args:
            new_features (pd.DataFrame): DataFrame containing the processed features.

        Returns:
            float: Probability of the transaction being fraudulent.
        """
        fraud_probability = self.model.predict_proba(new_features)[:, 1][0]
        # print(f"Likelihood of fraud: {fraud_probability * 100:.2f}%")
        return fraud_probability

    

if __name__ == '__main__':
    new_transaction = {
            "amount": 500.00,
            "transaction type": "cash withdrawal",
            "date": "2024-08-01 12:00:00",
            "ref": "capitec bank",
            "location": {"city": "cape town", "province": "gauteng"}
    }
    analysis = AnalyzeTransaction(new_transaction)
    features = analysis.process_transaction()
    analysis.predict_fraud(features)
    
    