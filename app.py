from flask import Flask, render_template, request
from preprocess import clean_text, load_model, load_vectorizer

app = Flask(__name__)
model = load_model()
vectorizer = load_vectorizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "Real News" if prediction == 1 else "Fake News"
    return render_template('index.html', prediction=label, original=news)

if __name__ == '__main__':
    app.run(debug=True)
