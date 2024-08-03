import joblib
import random
import pandas as pd
from datetime import datetime
from model import TransactionModel
from analyze import AnalyzeTransaction
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)


transactionModel = TransactionModel('transactions.json')
location = transactionModel.process_location()
amount = transactionModel.process_amount()
merchant = transactionModel.process_merchant()
X, y = transactionModel.create_feautures_dataframe(location, merchant, amount)
X_train, y_train = transactionModel.trainXandYsplit(X, y)
transactionModel.train_model(X_train, y_train)
transactionModel.save_model()


default_transaction = {"amount": 100.00, "trnsaction_type": "purchase", "ref": "mcdonald", "location": "city_province", "hour": 14, "day_of_week": 3}


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form.to_dict()
    try:
        transaction = {
            "amount": float(data.get('amount', 0)),
            "transaction type": data.get('transaction_type', ''),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ref": data.get('merchant_type', ''),
            "location": {"city": data.get('city', ''), "province": data.get('province', '')}
        }
        if (transaction['location']['province']=='gauteng'):
            transaction['location']['city'] = random.choice(['kaalfontein', 'Tembisa', 'ivory', 'joburg', 'cape town', 'rosebank', 'tsakane'])
        elif (transaction['location']['province']=='limpopo'):transaction['location']['city'] = 'moria'
        elif (transaction['location']['province']=='free state'):transaction['location']['city'] = 'qwaqwa'
        elif (transaction['location']['province']=='north west'):transaction['location']['city'] = 'mahikeng'
        elif (transaction['location']['province']=='western cape'):transaction['location']['city'] = 'cape town'
        elif (transaction['location']['province']=='eastern cape'):transaction['location']['city'] = 'gqeberha'
        elif (transaction['location']['province']=='kwazulu natal'):transaction['location']['city'] = 'vryheid'
        elif (transaction['location']['province']=='mpumalanga'):transaction['location']['city'] = 'bethal'
        
        analyzer = AnalyzeTransaction(transaction)
        features = analyzer.process_transaction()
        fraud_probability = analyzer.predict_fraud(features)
        response = {"fraud_probability": fraud_probability}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)