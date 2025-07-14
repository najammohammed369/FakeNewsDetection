# 📰 Fake News Detection Web App

This is a machine learning web application that detects whether a given news article or headline is **Fake** or **Real** using Natural Language Processing (NLP) techniques and a Logistic Regression model.

---

## 📂 Project Structure

fake-news-detector/
├── fakenews-dataset/ # Raw CSV files
│ ├── Fake.csv
│ └── True.csv
│
├── model/ # Saved model and vectorizer
│ ├── model.pkl
│ └── vectorizer.pkl
│
├── static/ # (Optional) For custom CSS/JS/images
├── templates/ # Flask HTML templates
│ └── index.html
│
├── app.py # Flask web application
├── preprocess.py # Text cleaning and model loader
├── detection_model.ipynb # Notebook to train and save the model
├── requirements.txt # Python dependencies


---

## 🚀 Features

- Clean and simple web interface using **Flask**
- Uses **TF-IDF vectorization** and **Logistic Regression**
- Trained on combined `Fake.csv` and `True.csv` datasets
- Predicts if the given news is **Real** or **Fake**
- Option to expand with CSV upload or confidence scoring

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
bash
git clone https://github.com/najammohammed369/FakeNewsDetectoin.git
cd FakeNewsDetection
```

Technologies Used
Python
Flask
Pandas, NumPy
Scikit-learn
NLTK
HTML/CSS (via templates/index.html)
