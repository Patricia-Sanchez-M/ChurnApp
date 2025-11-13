# 📊 ChurnApp  
### 🔍 Predicting Customer Churn with Interactive Dashboards (Dash + ML Models)

ChurnApp is a fully interactive web application for **exploring customer behaviour**, **analyzing churn**, and **predicting the probability that a client will leave a telecom company**.  

Built using **Python, Dash, Plotly, Scikit-learn** and deployed on **Render using Docker** 🚀.

---

## ✨ Features

### 📈 **1. Exploratory Data Analysis (EDA)**
- Interactive visualizations  
- Boxplots, histograms, categorical distributions  
- Auto-updating KPIs  
- Correlation matrix with heatmap  
- Clean UI based on `dash-bootstrap-components`

---

### 🤖 **2. Machine Learning Models**
Includes **three trained ML models**:

| Model | Icon | Capabilities |
|-------|------|--------------|
| Decision Tree | 🌳 | Fast, interpretable baseline model |
| Random Forest | 🌲 | Ensemble model with feature importances |
| Neural Network (MLP) | 🧠 | Deep pattern recognition |

For each model, the app displays:
- 📊 Confusion matrix with green/red transparency  
- 📉 Full performance metrics (Accuracy, Precision, Recall, F1, AUC)  
- 📈 Bar chart comparing all three models  
- ⭐ Feature importance charts (Random Forest)

---

### 🧪 **3. Data Preview & Model Predictions**
- Interactive table of the Telco Customer Churn dataset  
- Click a row → shows predictions of:
  - 🌳 Decision Tree  
  - 🌲 Random Forest  
  - 🧠 Neural Network  
- Real churn label included for comparison  
- Color-coded predictions + probability bars

---

### 🧮 **4. Custom Prediction Form**
Users can enter their own customer profile to generate predictions:

- Demographics 👤  
- Contract information 💼  
- Payment method 💳  
- Services selection 🌐  
- Charges 💰  

All three models return:
- Prediction label (CHURN / NO CHURN)
- Probability bar
- Color-coded insights

---

## 🛠️ Tech Stack

### **Backend / ML**
- Python 3.10  
- Pandas, NumPy  
- Scikit-learn  
- Joblib  

### **Frontend (Dash)**
- Dash  
- Dash Bootstrap Components  
- Plotly  
- Figure Factory  

### **Deployment**
- Docker  
- Render Web Service  
- Gunicorn  

---

## 🚀 Running Locally

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/Patricia-Sanchez-M/ChurnApp.git
cd ChurnApp
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app
```bash
python app.py
```

### 🐳 Running with Docker
```bash
docker build -t churnapp .
docker run -p 8050:8050 churnapp
```

### 📁 Project Structure
```bash
ChurnApp/
│── app.py                 # Main Dash app
│── Dockerfile             # Deployment config
│── requirements.txt       # Dependencies
│── assets/                # CSS / custom styles
│── models/                # ML models + metrics + confusion matrices
│── pages/                 # Multipage Dash routes
│    ├── home.py
│    ├── eda.py
│    ├── data.py
│    ├── models.py
│    └── predict.py
│── Telco-Customer-Churn.csv
```

### 
🌐 Live Demo
```bash
🚀 https://churnapp-1.onrender.com
```