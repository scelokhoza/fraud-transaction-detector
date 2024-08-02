import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




class TransactionModel:
    """
    A class to preprocess transaction data, train a RandomForest model, and save the model along with necessary encoders.

    Attributes:
        file (str): Path to the transaction data JSON file.
        transaction_data (list): List of transaction dictionaries loaded from the JSON file.
        data (pd.DataFrame): DataFrame containing all transaction data.
        user_data (pd.DataFrame): DataFrame containing data specific to a particular user.
        location_encoder (OneHotEncoder): Encoder for transforming location data.
        merchant_encoder (OneHotEncoder): Encoder for transforming merchant data.
        scaler (StandardScaler): Scaler for normalizing transaction amounts.
        model (RandomForestClassifier): The machine learning model used for fraud detection.

    Methods:
        open_file() -> list[dict, dict]: Opens and reads the transaction data from the JSON file.
        preprocess_data(): Preprocesses the transaction data, extracting and engineering features.
        process_location() -> np.ndarray: Encodes the location feature.
        process_merchant() -> np.ndarray: Encodes the merchant feature.
        process_amount() -> np.ndarray: Scales the transaction amount feature.
        create_feautures_dataframe(location_encoded, merchant_encoded, amount_scaled) -> tuple: Creates a features DataFrame for model training.
        trainXandYsplit(X, y) -> tuple: Splits the data into training and test sets.
        train_model(X_train, y_train): Trains the RandomForest model.
        save_model(): Saves the trained model and encoders using joblib.
    """

    def __init__(self, file: str) -> None:
        """
        Initializes the TransactionModel with the given file path.

        Args:
            file (str): The file path to the transaction data JSON file.
        """
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
        """
        Opens and reads the transaction data from the JSON file.

        Returns:
            list[dict, dict]: A list of transaction data dictionaries.
        """
        with open(self.file, 'r') as file:
            transaction_data = json.load(file)
        return transaction_data

    def preprocess_data(self):
        """
        Preprocesses the transaction data, extracting and engineering features such as date, hour, and location.
        """
        self.user_data['date'] = pd.to_datetime(self.user_data['date'])
        self.user_data['hour'] = self.user_data['date'].dt.hour
        self.user_data['day_of_week'] = self.user_data['date'].dt.dayofweek
        self.user_data['location'] = self.user_data['location'].apply(lambda x: f"{x['city']}_{x['province']}")

    def process_location(self) -> np.ndarray:
        """
        Encodes the location feature using OneHotEncoder.

        Returns:
            np.ndarray: Encoded location data.
        """
        location_encoded = self.location_encoder.fit_transform(self.user_data[['location']])
        return location_encoded

    def process_merchant(self) -> np.ndarray:
        """
        Encodes the merchant feature using OneHotEncoder.

        Returns:
            np.ndarray: Encoded merchant data.
        """
        merchant_encoded = self.merchant_encoder.fit_transform(self.user_data[['ref']])
        return merchant_encoded

    def process_amount(self) -> np.ndarray:
        """
        Scales the transaction amount feature using StandardScaler.

        Returns:
            np.ndarray: Scaled transaction amounts.
        """
        amount_scaled = self.scaler.fit_transform(self.user_data[['amount']])
        return amount_scaled

    def create_feautures_dataframe(self, location_encoded, merchant_encoded, amount_scaled) -> tuple:
        """
        Creates a DataFrame with encoded and scaled features for model training.

        Args:
            location_encoded (np.ndarray): Encoded location data.
            merchant_encoded (np.ndarray): Encoded merchant data.
            amount_scaled (np.ndarray): Scaled transaction amounts.

        Returns:
            tuple: Tuple containing features DataFrame (X) and target series (y).
        """
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
        return X, y

    def trainXandYsplit(self, X, y) -> tuple:
        """
        Splits the data into training and test sets.

        Args:
            X (pd.DataFrame): Features DataFrame.
            y (pd.Series): Target series.

        Returns:
            tuple: Tuple containing training features (X_train) and training target (y_train).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, y_train

    def train_model(self, X_train, y_train):
        """
        Trains the RandomForest model on the training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        self.model.fit(X_train, y_train)

    def save_model(self):
        """
        Saves the trained model and encoders using joblib.
        """
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