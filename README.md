# 😊 Emotion Detection in Text - Social Media Analysis

A powerful web application for detecting emotions in social media text posts using Natural Language Processing and Machine Learning. This app can analyze individual texts or process multiple texts in batch, providing detailed emotion classification with confidence scores.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Screenshots](#screenshots)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Introduction

Emotion detection in text data involves identifying the emotions expressed in textual content. This can be a challenging task since emotions are often expressed in complex and subtle ways. This application uses Natural Language Processing (NLP) techniques and Machine Learning to analyze text data and accurately identify emotions.

The app is designed for:
- **Social media sentiment analysis**
- **Customer feedback analysis**
- **Market research and consumer insights**
- **Brand monitoring and reputation management**
- **Content analysis and recommendation systems**

## ✨ Features

### 🏠 Single Text Analysis
- **Real-time emotion detection** - Instantly analyze emotions in any text
- **Interactive visualizations** - Beautiful charts showing emotion probabilities
- **Confidence scores** - See how confident the model is in its predictions
- **Example texts** - Quick buttons to try sample texts
- **Color-coded emotions** - Visual representation with emojis and colors

### 📊 Batch Text Analysis (NEW!)
- **Multiple text processing** - Analyze hundreds of texts at once
- **Comprehensive results table** - View all predictions in one place
- **Statistics dashboard** - Total texts, unique emotions, average confidence
- **Emotion distribution charts** - Visual breakdown of detected emotions
- **CSV export** - Download results for further analysis

### 📈 Analytics & Monitoring
- **Page visit tracking** - Monitor app usage statistics
- **Prediction history** - View all past predictions
- **Time series analysis** - Track predictions over time
- **Export functionality** - Download all data as CSV
- **Interactive dashboards** - Beautiful charts and visualizations

### 🎨 Modern UI
- **Beautiful gradient design** - Modern and professional interface
- **Responsive layout** - Works on all screen sizes
- **Interactive charts** - Powered by Plotly for better visualization
- **User-friendly** - Intuitive navigation and clear results

## 📸 Screenshots

### Home Page - Single Text Analysis
The main page allows you to analyze individual texts with detailed emotion breakdowns and confidence scores.

### Batch Analysis Page
Process multiple texts simultaneously and export results as CSV.

### Monitor Dashboard
Track all predictions, view statistics, and analyze trends over time.

## 📊 Dataset

The dataset used for this project contains text data labeled with emotions:
- **Anger** 😠
- **Disgust** 🤮
- **Fear** 😨
- **Happy** 🤗
- **Joy** 😂
- **Neutral** 😐
- **Sad/Sadness** 😔
- **Shame** 😳
- **Surprise** 😮

**Total Dataset Size:** 34,795 rows

The dataset is located in the `data/` directory.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aryan-07-web/social-media-text-emotion-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Model File

Ensure the trained model file exists at:
```
models/emotion_classifier_pipe_lr.pkl
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

### Using Single Text Analysis

1. Navigate to the **Home** page
2. Enter your text in the text area (or click an example button)
3. Click **Analyze Emotion**
4. View the detected emotion, confidence score, and probability distribution

### Using Batch Analysis

1. Navigate to the **Batch Analysis** page
2. Enter multiple texts (one per line)
3. Click **Analyze All**
4. View results in the table
5. Export results as CSV if needed

### Viewing Analytics

1. Navigate to the **Monitor** page
2. View page visit metrics and prediction history
3. Analyze trends and export data

## 🛠️ Technologies Used

- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn (Logistic Regression)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Altair
- **Database:** SQLite
- **NLP:** Custom preprocessing pipeline with NeatText

## 📈 Model Performance

- **Model:** Logistic Regression
- **Accuracy:** 62%
- **Supported Emotions:** 10 different emotion classes


