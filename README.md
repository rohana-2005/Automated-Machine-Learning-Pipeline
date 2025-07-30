# Automated Machine Learning Pipeline 🚀

This project is a web-based **Automated Machine Learning (AutoML) Pipeline** built with **Flask**. It allows users to upload CSV datasets, automatically preprocess them, train multiple machine learning models (classification or regression), compare model performance, and download the best-performing model—all through a simple web interface.

---

## 🔧 Features

- 📁 Upload CSV data via UI
- 🧹 Automated data cleaning:
  - Remove duplicates
  - Handle missing values
  - Outlier removal (IQR method)
  - Standardize numeric columns
  - Encode categorical variables
- 🔍 Choose problem type: Classification or Regression
- ⚙️ Train multiple models (e.g., Decision Trees, Random Forests, SVM, Linear/Logistic Regression)
- 📊 Compare model performance (Accuracy, MSE, etc.)
- 💾 Download trained models as `.pkl` files
- 🖥️ Predict outcomes with the selected model via UI

---

## 🚀 Getting Started

### 1. Clone the repository and set up the environment
```bash
git clone https://github.com/rohana-2005/Automated-Machine-Learning-Pipeline.git
cd Automated-Machine-Learning-Pipeline

python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
python app.py

---

🧰 Tech Stack
Python
Flask (Backend)
HTML/CSS (Frontend)
Scikit-learn (ML Models & Evaluation)
Pandas & NumPy (Data Processing)
