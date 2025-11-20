# 📊 ChurnApp

### 🔍 Predicting Customer Churn with Interactive Dashboards (Dash + ML Models)

ChurnApp is a fully interactive web application for **exploring customer behaviour**, **analyzing churn**, and **predicting the probability that a client will leave a telco company**.

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

| Model                | Icon | Capabilities                            |
| -------------------- | ---- | --------------------------------------- |
| Decision Tree        | 🌳   | Fast, interpretable baseline model      |
| Random Forest        | 🌲   | Ensemble model with feature importances |
| Neural Network (MLP) | 🧠   | Deep pattern recognition                |

For each model, the app displays:

- 📊 Confusion matrix with green/red transparency
- 📈 Full performance metrics (Accuracy, Precision, Recall, F1, AUC)
- ⭐ Feature importance charts (Random Forest)

#### 🧠 Model training

All models are **binary classifiers** that predict whether a customer will **churn (`Yes`) or not churn (`No`)**.  
Training is done **offline** in the Jupyter notebooks included in this repo: **[ChurnApp Models](https://github.com/Patricia-Sanchez-M/ChurnApp_Models)**

- `churn_EDA.ipynb` – Loads the Telco Customer Churn dataset, cleans basic issues
  (`TotalCharges` as numeric, `Churn` mapped to 0/1) and explores the main patterns
  that later guide the choice of features and models.

##### 🌳 Decision Tree (CART)

Notebook: `churn_CART.ipynb`

- Model: `sklearn.tree.DecisionTreeClassifier` trained on the cleaned dataset.
- The tree is **regularised and pruned** to avoid overfitting:
  - First a full tree is fitted and the **cost-complexity pruning path** is computed.
  - Using `ccp_alpha` values from this path, several pruned trees are evaluated.
  - The final `ccp_alpha` (and depth / min samples per leaf) is chosen as a compromise
    between **simplicity** (shallower tree, fewer leaves) and **validation performance**.
- Metrics (Accuracy, Precision, Recall, F1, AUC) and the confusion matrix are saved as
  CSV/JSON in `models/DecisionTreeClassifier_*.{csv,json}`.
- In the app, the **churn probability** is the **class proportion in the leaf**
  where the customer ends up (leaf purity).
  - Example: if 80% of training samples in that leaf were churners → `P(churn) = 0.8`.

##### 🌲 Random Forest

Notebook: `churn_ensembles_RandomForest.ipynb`

- Model: `sklearn.ensemble.RandomForestClassifier` trained on the same feature set.
- Key hyperparameters tuned/checked:
  - `n_estimators` (number of trees),
  - `max_depth` / `min_samples_leaf` (to control overfitting),
  - `max_features` (features considered at each split),
  - `class_weight` (to reduce imbalance between churn / no churn if needed).
- The forest aggregates predictions by **averaging the probabilities** of all trees
  and predicting the class with the highest probability.
- Evaluation metrics, confusion matrix and **feature importances** are exported to the
  `models/` folder.

##### 🧠 Neural Network (MLP)

Notebook: `churn_MLP.ipynb`

- Model: `sklearn.neural_network.MLPClassifier` wrapped in a `Pipeline`:
  - `ColumnTransformer` for preprocessing:
    - numeric features → `MinMaxScaler`,
    - categorical features → `OneHotEncoder`.
  - The transformed features feed the MLP classifier.
- The network is trained as a **probabilistic classifier**:
  - the final neuron outputs `P(churn)` via a sigmoid/softmax layer.
- Relevant hyperparameters (hidden layer sizes, activation, regularisation and
  early-stopping settings) are chosen based on validation performance, not just
  training accuracy.
- The fitted **preprocessor** and **MLP model** are saved with `joblib` in `models/`.

###### 🔮 How the app uses the models

The Dash app **does not re-train** models in production. It simply:

1. Loads the serialized models and preprocessing objects from `models/`  
   (`churn_*Classifier.pkl`, `*_columns.pkl`, `churn_MLPClassifier_preprocessor.pkl`).
2. Builds a feature vector from the user input (or selected row in the dataset).
3. Asks each model for the **churn probability**:
   - Decision Tree → leaf purity,
   - Random Forest → average probability across trees,
   - MLP → neural network output.
4. In the **`/predict`** page, these probabilities are converted to:
   - a **label** (`CHURN` / `NO CHURN`, using a 0.5 threshold), and
   - a **progress bar** that visually encodes `P(churn)` for each model card.

---

### 🧪 **3. Data Preview & Model Predictions**

- Interactive table of the Telco Customer Churn dataset
- Click a row → shows predictions of:
  - 🌳 Decision Tree
  - 🌲 Random Forest
  - 🧠 Neural Network
- Real churn label included for comparison

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

---

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

---

### 🌐 Live Demo

**[ChurnApp 🚀](https://churnapp-4ik3.onrender.com/predict)**  
_Cold starts may take 30–50 seconds._
