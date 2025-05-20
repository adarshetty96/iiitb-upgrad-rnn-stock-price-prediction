# 📈 RNN-Based Stock Price Prediction

This repository contains a time series forecasting project using **Recurrent Neural Networks (RNNs)** to predict stock prices. It explores both **simple and advanced RNN architectures** (LSTM and GRU), and evaluates their performance on **single** and **multi-target** stock prediction tasks.

Developed as part of the **IIITB-UpGrad PG Diploma in Data Science** program.

---

## 🧠 Project Overview

This project aims to build deep learning models to forecast the closing price of stocks using historical data. Two core scenarios were considered:

1. **Single Stock Prediction** – Predicting closing prices for a single stock (AMZN).
2. **Multi-Stock Prediction** – Simultaneous prediction of multiple stock closing prices.

We implemented:
- Simple RNN models
- Advanced RNN models using **LSTM** and **GRU**
- Hyperparameter tuning for optimal performance

---

## 📊 Dataset

- **Source**: [Add dataset link if from Kaggle or elsewhere]
- **Features**: Date, Open, High, Low, Close, Volume
- **Preprocessing**: Normalization, Windowing (for sequence modeling), Train/Val/Test split

---

## 🏗️ Model Architectures

### 1. Simple RNN
- Units: 64–128
- Activation: `relu` / `tanh`
- Dropout: 0.1 – 0.3

### 2. Advanced RNN
- **LSTM** and **GRU**
- Tuned parameters: units, dropout, activation

---

## 🔧 Hyperparameter Tuning

| Model Type       | Units | Dropout | Activation | Notes                         |
|------------------|-------|---------|------------|-------------------------------|
| Simple RNN (Single) | 64    | 0.1     | relu       | Best for AMZN Close Price     |
| LSTM (Single)       | 64    | 0.2     | relu       | Best advanced model           |
| Simple RNN (Multi)  | 128   | 0.3     | tanh       | Best for multi-target         |
| GRU (Multi)         | 128   | 0.3     | tanh       | Best overall for multi-target |

---

## 📈 Results

### 🟦 Single Stock (AMZN Close Price)

#### 🔹 Simple RNN (Best Config: 64 units, 0.1 dropout, relu)
- **Validation Performance**:
  - MAE  : 0.0749
  - MSE  : 0.0111
  - RMSE : 0.1052
  - R²   : 0.9889
- **Test Performance**:
  - MAE  : 0.0749
  - MSE  : 0.0111
  - RMSE : 0.1052
  - R²   : 0.9889

#### 🔹 Advanced RNN (LSTM: 64 units, 0.2 dropout, relu)
- **Validation Loss**: 0.00908
- **Test Loss**: 0.01564
- **Test Metrics**:
  - MAE  : 0.0971
  - MSE  : 0.0156
  - RMSE : 0.1251
  - R²   : 0.9844

---

### 🟨 Multi-Stock Prediction (Window size: 65, Stride: 5)

#### 🔹 Simple RNN (128 units, 0.3 dropout, tanh)
- **Test Loss**: 0.0449
- **MAE**: 0.1726

#### 🔹 Advanced RNN (GRU: 128 units, 0.3 dropout, tanh)
- **Test Loss**: 0.0390
- **MAE**: 0.1599

---

## 🧠 Key Insights and Observations

- **RNN architectures** perform strongly for time series forecasting with appropriate tuning.
- **LSTM** and **GRU** outperform simple RNNs in most scenarios due to better long-term memory handling.
- For **multi-stock prediction**, **GRU** models provided the best trade-off between performance and complexity.
- Increasing **window size and stride** improves sequence learning for multi-target scenarios but also increases training time.

---

## ✅ Final Outcomes

- Accurate stock price prediction models with **R² scores > 0.98** for AMZN.
- Successfully implemented **multi-target** forecasting using advanced RNN models.
- Delivered scalable and tunable models for both individual and group stock price forecasting.

---

## 🏁 Conclusion

This project demonstrates the power of RNN-based models, especially **LSTM** and **GRU**, for time series forecasting. Through careful tuning and comparative evaluation, we have developed models capable of accurately predicting stock closing prices in both single and multi-target settings.

---

## ▶️ How to Run

1. Clone this repository  
   ```bash
   git clone https://github.com/adarshetty96/iiitb-upgrad-rnn-stock-price-prediction.git
   cd iiitb-upgrad-rnn-stock-price-prediction

2.Install dependencies
pip install -r requirements.txt

3.Run the notebooks in Jupyter Lab or Jupyter Notebook.

Requirements
Python 3.11
TensorFlow / Keras
NumPy, Pandas, Matplotlib, Seaborn
scikit-learn

📚 Acknowledgments
IIIT Bangalore & UpGrad PG Diploma in Data Science
