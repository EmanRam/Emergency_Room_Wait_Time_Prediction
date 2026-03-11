# 🏥 ER Wait Time Predictor

A machine learning web application that predicts **Emergency Room wait times** based on patient, hospital, and contextual factors. Built with **Streamlit** and trained on 5,000 real ER visit records, the app helps hospital staff and administrators anticipate congestion and improve patient flow management.

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Usage Guide](#-usage-guide)
- [Contributing](#-contributing)

---

## 📌 Project Description

Emergency rooms face unpredictable patient surges, making it difficult to allocate resources efficiently. Long wait times directly impact patient outcomes and satisfaction.

This project addresses that challenge by training regression models on historical ER visit data to **forecast total patient wait time in minutes**. The result is an interactive dashboard where users can input patient and hospital parameters and receive an instant, color-coded wait time prediction.

The app uses the best model which is **Gradient Boosting** to make prediction.

---

## ✨ Features

- 🤖 **ML Models** — Gradient Boosting regression
- ⚡ **Instant Predictions** — Fill in patient/hospital details and get a wait time estimate in under a second
- 📊 **Exploratory Dashboard** — Summary statistics and visual breakdowns of the ER dataset
- 🎨 **Color-coded Results** — Green / yellow / red indicators based on estimated wait severity
- 💾 **Pre-trained Model Support** — Loads a saved `.pkl` artifact automatically; retrains on the fly if not found
- 📓 **Jupyter Notebooks** — Full EDA and model development workflow included

---

## 📂 Dataset

| Property | Details |
|---|---|
| **File** | `ER Wait Time Dataset.csv` |
| **Records** | 5,000 ER visits |
| **Target Variable** | `Total Wait Time (min)` |

**Features include:**

- `Visit Date`, `Day of Week`, `Season`, `Time of Day`
- `Urgency Level`, `Nurse-to-Patient Ratio`, `Specialist Availability`
- `Facility Size (Beds)`, `Region`, `Hospital Name`

---

## 🗂 Project Structure

```
ER_Wait_Time/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── ER Wait Time Dataset.csv      # Source dataset (5,000 records)
├── er_wait_model.pkl             # Saved best model artifact (generated after running notebook)
│
└── notebooks/
    ├── Hospital_ER_Wait_Time.ipynb   # Exploratory Data Analysis
    └── ER_Wait_Time_3.ipynb          # Model training, evaluation & saving
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/er-wait-time-predictor.git
cd er-wait-time-predictor
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

```bash
# Or using conda
conda create -n er-wait python=3.10
conda activate er-wait
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥 Usage Guide

1. Launch the app with `streamlit run app.py`
2. On the **Home** page, review the dataset overview and how-it-works breakdown
3. Navigate to the **Predict** page using the sidebar
4. Fill in the form:
   - Hospital details (region, facility size, staffing)
   - Visit context (time of day, day of week, season)
   - Patient details (urgency level, registration/triage times)
5. Select a model and click **Predict**
6. View your color-coded wait time estimate instantly

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure any new code follows the existing style and includes relevant comments.

---

*Built with ❤️ using Python, scikit-learn, and Streamlit.*
