# 🧠 Real-Time Fake News Detection using Machine Learning

<p align="center">
<img src="screenshots/project-banner.png" width="850">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

A **Real-Time Fake News Detection System** built using **Machine Learning, Natural Language Processing (NLP), and AI semantic similarity analysis**.

The application analyzes news text and predicts whether it is **Fake or Real**, verifies it using **live internet news search**, and evaluates **semantic similarity using AI models**.

The project includes an **interactive Streamlit dashboard** where users can test news articles in real time.

---

# 📌 Project Overview

Fake news spreads rapidly across social media and digital platforms, making it difficult to identify reliable information.

This project aims to:

* Detect fake news using **Machine Learning**
* Process text using **Natural Language Processing**
* Verify news using **live internet search**
* Measure semantic similarity using **AI embeddings**
* Provide an **interactive web interface**

---

# 🎬 Project Demo

<p align="center">
<img src="screenshots/demo.gif" width="750">
</p>

The demo demonstrates:

* Fake vs Real news prediction
* Prediction confidence score
* Real-time internet news verification
* AI similarity checking
* Interactive Streamlit dashboard

---

# 🚀 Features

* Fake vs Real news classification
* Natural Language Processing (NLP)
* TF-IDF feature extraction
* Logistic Regression ML model
* Prediction confidence score
* Live news search using News API
* AI semantic similarity verification
* Interactive Streamlit web interface
* Model accuracy dashboard

---

# 🛠 Technologies Used

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Programming Language      |
| Streamlit             | Web Application Framework |
| Scikit-Learn          | Machine Learning          |
| Pandas                | Data Processing           |
| Sentence Transformers | Semantic Similarity       |
| Requests              | API Integration           |
| NLP                   | Text Processing           |

---

# 🧠 Machine Learning Model

The system uses **Logistic Regression** trained on a dataset containing fake and real news articles.

### Feature Extraction

TF-IDF (**Term Frequency – Inverse Document Frequency**)

TF-IDF converts textual data into numerical vectors so machine learning models can process them.

---

# ⚙ Model Pipeline

```
News Input
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Machine Learning Model
   ↓
Prediction (Fake / Real)
   ↓
Confidence Score
   ↓
Live News Verification
   ↓
AI Similarity Check
```

---

# 📊 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~96%     |

Accuracy may vary depending on preprocessing and dataset sampling.

---

# 📂 Project Structure

```
fake-news-detector
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset
│   ├── Fake.csv
│   └── True.csv
│
├── screenshots
│   ├── dashboard.png
│   ├── prediction.png
│   ├── demo.gif
│   └── project-banner.png
│
├── demo
│   └── project-demo.mp4
```

---

# 📦 Dataset

This project uses the **Fake and Real News Dataset**.

Because the dataset files are large, they are **not uploaded to GitHub**.

### Download Dataset

Download from:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Required Files

```
Fake.csv
True.csv
```

### Place Files In

```
dataset/Fake.csv
dataset/True.csv
```

### Example Code Path

```python
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")
```

---

# ⚙ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/yourusername/fake-news-detector.git
```

### 2️⃣ Navigate to Project Folder

```
cd fake-news-detector
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

# 📦 requirements.txt

```
streamlit
pandas
scikit-learn
requests
sentence-transformers
```

---

# 🖥 Application Screenshots

### Dashboard

<img width="943" height="502" alt="image" src="https://github.com/user-attachments/assets/c6c41243-b4a9-4446-8b71-14c51056bffb" />


### Prediction Example

<img width="955" height="498" alt="image" src="https://github.com/user-attachments/assets/061c3331-49e1-4d6c-aee7-935342fc2cef" />

<img width="960" height="502" alt="image" src="https://github.com/user-attachments/assets/65fa6e15-0b29-47c6-8e19-0386cf4ab60b" />



---
# 🎥 Project Demo

You can watch the **full working demo of this project** from the link below:

➡ **[Watch Demo Video](Demo.mp4)**

The demo video includes:

* Project overview
* Fake vs Real news prediction
* Prediction confidence score
* Live internet news verification
* AI similarity checking
* Streamlit dashboard interface

---

### 📁 Demo Video Location

The video is stored inside the project folder:

```
demo/
└── project-demo.mp4
```






# 📈 Example Output

```
ML Prediction: Real News

Prediction Confidence: 92%

Similarity Score: 0.74

Similar news found online → Likely Real
```

---

# ⚠ Limitations

* Fake news detection cannot be **100% accurate**
* Predictions depend on **training dataset patterns**
* Real-time verification depends on **external APIs**

---

# 🔮 Future Improvements

* Integrate **Deep Learning models (BERT)**
* Improve real-time fact checking
* Add interactive data visualizations
* Deploy the project online
* Add multilingual support

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Submit a pull request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

Developed as a **Machine Learning and NLP project for detecting fake news using AI**.

---

⭐ If you find this project useful, consider **starring the repository**.
