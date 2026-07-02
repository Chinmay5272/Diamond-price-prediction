# 💎 Diamond Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts the price of diamonds using Machine Learning regression models. Multiple regression algorithms were trained and compared, and the best-performing model was further optimized using GridSearchCV to improve prediction accuracy.

---

## 🎯 Objective

The objective of this project is to accurately predict diamond prices based on their physical and quality characteristics while comparing multiple machine learning models.

---

## 📂 Dataset

The dataset contains information about diamonds, including:

- Carat
- Cut
- Color
- Clarity
- Depth
- Table
- X
- Y
- Z

**Target Variable**

- Price

---

## ⚙️ Data Preprocessing

- Removed unnecessary columns
- One-Hot Encoding
- Train-Test Split
- Feature Scaling (for applicable models)

---

## 🤖 Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors (KNN)
- Gradient Boosting Regressor
- XGBoost Regressor

---

## 🔧 Hyperparameter Tuning

The best-performing model (XGBoost) was optimized using **GridSearchCV** to improve prediction performance.

---

## 📊 Model Performance

| Model | R² Score |
|--------|----------|
| Linear Regression | 0.859 |
| Decision Tree | 0.778 |
| Random Forest | 0.880 |
| Gradient Boosting | 0.886 |
| KNN | 0.868 |
| XGBoost (Default) | 0.887 |
| **XGBoost (GridSearchCV)** | **0.980** |

---

## 📈 Visualizations

- Feature Importance
- Actual vs Predicted Prices
- Residual Analysis
- Model Comparison

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- Joblib
- Jupyter Notebook

---

## 📁 Project Structure

```text
Diamond-Price-Prediction/
│
├── Diamond_Price_Prediction.ipynb
├── diamonds.csv
├── diamond_price_model.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Results

- Best Model: **XGBoost Regressor**
- Hyperparameter Tuning: **GridSearchCV**
- Final R² Score: **0.980**
- MAE: **289.30**
- RMSE: **565.17**

---

## 📌 Future Improvements

- Deploy using Streamlit or Flask
- Build a web interface for predictions
- Experiment with advanced ensemble methods

---

