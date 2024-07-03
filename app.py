from flask import Flask, request, make_response
import numpy as np
import pandas as pd
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


app = Flask(__name__)
CORS(app)

@app.route('/process-csv', methods=['POST'])
def process_csv():
    # Retrieve the uploaded file from the request
    file = request.files['file']

    # Save the uploaded file to disk
    file_path = 'uploaded_file.csv'
    file.save(file_path)

    df = pd.read_csv(file_path)

    # Drop all rows with NaN values
    df = df.dropna()

    # Split the data into features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    df_without_fraud = df[df['Class'] == 0]
    df_without_fraud.to_csv('withoutcreditcards.csv', index=False)

    # Create a response with the CSV file data
    response = make_response(df_without_fraud.to_csv(index=False))
    response.headers['Content-Disposition'] = 'attachment; filename=withoutcreditcards.csv'
    response.headers['Content-Type'] = 'text/csv'

    return response


 
if __name__ == '__main__':
    app.run()

