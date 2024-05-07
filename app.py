import os
import pickle
import flask
import urllib
import newspaper
from newspaper import Article
import nltk
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model

nltk.download('punkt')

# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='template')

# Load the TF-IDF Vectorizer
with open('vectorizer_model.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

# Load the PAC model
with open('pac.pkl', 'rb') as pac_file:
    pac_model = pickle.load(pac_file)

# Load the ANN-BiLSTM model
#ann_bilstm_model = load_model('ann_model.h5')

@app.route('/', methods=['GET'])
def page():
    return render_template('fake.html')

# Receiving the input url from the user and using Web Scr
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_data = request.form['news']
    input_data = urllib.parse.unquote(input_data)

    # Check if the input is a URL
    if urllib.parse.urlparse(input_data).scheme in ['http', 'https']:
        # If it's a URL, download the article and extract the summary
        article = Article(str(input_data))
        article.download()
        article.parse()
        article.nlp()
        news = article.summary
    else:
        # If it's not a URL, assume it's plain text and use it directly
        news = input_data

    # Vectorize the news text
    tfidf_data = tfidf_vectorizer.transform([news])

    # Make predictions using the PAC model
    pac_prediction = pac_model.predict(tfidf_data)
    prediction_text_pac = 'FAKE' if pac_prediction[0] == 0 else 'REAL'

    # Make predictions using the ANN-BiLSTM model
    #ann_bilstm_prediction = ann_bilstm_model.predict(tfidf_data)
    #prediction_text_ann= 'FAKE' if ann_bilstm_prediction[0] == 0 else 'REAL'

    return render_template('fake.html', prediction_text=f'The News is "{prediction_text_pac}"')

@app.route('/predict/', methods=['GET', 'POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)