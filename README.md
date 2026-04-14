# Customer-Churn-Predection_ANN-keras
#### AI — Customer Churn Prediction

> **Predict customer churn risk with deep learning precision.**  
> An ANN-powered web application built with TensorFlow and Streamlit, deployed on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/mad0030/Churn_Pred_ANN)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Overview

**ChurnSight AI** is an end-to-end machine learning application that predicts the probability of a customer churning based on key account features. It uses a trained **Artificial Neural Network (ANN)** model to output a real-time churn risk score, categorizing customers into **High Risk** 🔴 or **Low Risk** 🟢 tiers — empowering businesses to act proactively on retention.

---

## 🚀 Live Demo

👉 Try it now on Hugging Face Spaces: [ChurnSight AI](https://huggingface.co/spaces/mad0030/Churn_Pred_ANN)

---

## ✨ Features

- ⚡ **Real-time churn probability prediction** using a trained Keras ANN model
- 📊 **Visual risk result card** with color-coded High/Low risk tiers
- 🎛️ **Interactive input controls** — sliders, dropdowns, and number inputs
- 📈 **Churn probability progress bar** for intuitive risk visualization
- 🧾 **Input summary chips** for a quick at-a-glance review
- 🌙 **Dark-themed, modern UI** built with custom CSS and Google Fonts
- ☁️ **Deployed on Hugging Face Spaces** — zero setup required

---

## 🧠 Model Details

| Attribute         | Details                            |
|-------------------|------------------------------------|
| Model Type        | Artificial Neural Network (ANN)    |
| Framework         | TensorFlow / Keras                 |
| Input Features    | Tenure, Monthly Charges, Contract Type, Internet Service |
| Output            | Churn Probability (0.0 – 1.0)      |
| Threshold         | > 0.5 → High Risk; ≤ 0.5 → Low Risk |
| Preprocessing     | StandardScaler (saved via joblib)  |
| Model Format      | `.keras`                           |

---

## 🗂️ Project Structure

<img width="1433" height="851" alt="image" src="https://github.com/user-attachments/assets/e0e3e48f-2218-4bdf-91d9-e3cc33a3e062" />


---

## 🖥️ Input Features

| Feature             | Type        | Description                                   |
|---------------------|-------------|-----------------------------------------------|
| **Tenure**          | Integer     | Number of months the customer has been active |
| **Monthly Charges** | Float       | Customer's current monthly billing amount ($) |
| **Contract Type**   | Categorical | Month-to-month / One year / Two year          |
| **Internet Service**| Categorical | DSL / Fiber optic / No                        |

---

## ⚙️ Installation & Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Churn_Pred_ANN.git
cd Churn_Pred_ANN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

> Make sure the `Models/` folder with `final_model.keras` and `final_scaler.pkl` is present before running.

---

## 📦 Dependencies
streamlit

tensorflow

numpy

joblib

scikit-learn


---

## 📷 Screenshots

> <img width="621" height="991" alt="image" src="https://github.com/user-attachments/assets/d2e8bcbe-7ac9-4c24-87f1-e927d4b4ab26" />


---

## 🛠️ How It Works

1. The user enters customer details via the interactive form.
2. Inputs are encoded (categorical → numeric) and scaled using the saved `StandardScaler`.
3. The preprocessed features are passed to the trained ANN model.
4. The model outputs a churn probability score between 0 and 1.
5. The app displays the result as a styled **Risk Card** with an actionable message.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 👤 Author

**Madhava Reddy Yanamala**  
🤗 [Hugging Face Profile](https://huggingface.co/mad0030)

---

> *ChurnSight AI · Built with Streamlit + TensorFlow · © 2025*
