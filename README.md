# 🏥 ER Wait Time Predictor

A machine learning web application that predicts **Emergency Room (ER) wait times** using a **final no-leak Gradient Boosting model** trained on **Dataset 3**. Built with **Streamlit** and trained on **5,000 ER visit records**, the app helps estimate total patient wait time based on hospital, staffing, urgency, and visit timing factors.

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Key Findings](#-key-findings)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Usage Guide](#-usage-guide)
- [Contributing](#-contributing)

---

## 📌 Project Description

Emergency rooms face unpredictable patient surges, making it difficult to allocate resources efficiently. Long wait times affect patient satisfaction, delay treatment, and create operational pressure on hospitals.

This project addresses that challenge by training regression models on historical ER visit data to **predict total wait time in minutes**. The final deployed system is based on the strongest **trustworthy** model from the project: a **Gradient Boosting Regressor** trained on the **Dataset 3 no-leak feature set**.

The web app provides an interactive interface where users can enter operational ER conditions and receive an instant, color-coded estimate of expected wait time.

---

## 🔍 Key Findings

This project followed a two-phase workflow:

### Phase 1 — Baseline Patient-Level Dataset
A patient-level dataset containing demographic and administrative variables was used as the baseline.

**Result:**
- Very weak predictive performance
- Best validation score was approximately **R² = 0.008**
- Test performance remained negative

**Insight:**  
Patient demographics and basic administrative features alone were not sufficient to explain ER wait time.

### Phase 2 — Operational Dataset
A richer dataset containing operational features such as urgency, staffing, hospital size, and visit timing was then used.

Initial results were nearly perfect, but this was caused by **target leakage**.

#### Leaked features removed:
- `Time to Registration`
- `Time to Triage`
- `Time to Medical Professional`

After removing leaked variables, the final model became realistic and trustworthy.

---

## ✨ Features

- 🤖 **Final Deployed Model** — Gradient Boosting Regressor (No-Leak)
- ⚡ **Instant Predictions** — Get an ER wait time estimate in seconds
- 🧠 **Operational Inputs** — Uses hospital, staffing, urgency, and visit timing variables
- 🎨 **Color-coded Results** — Green / amber / red severity levels
- 💾 **Pre-trained Model Support** — Loads a saved `.pkl` artifact from the `models/` folder
- 📓 **Notebook Workflow** — Includes model development and exploratory analysis notebooks
- 🗂 **Clean Repo Structure** — Organized into `data/`, `models/`, and `notebooks/`

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Dataset** | ER Wait Time Dataset |
| **Records** | 5,000 ER visits |
| **Target Variable** | `Total Wait Time (min)` |

### Main input features used by the app
- `Hospital Name`
- `Region`
- `Day of Week`
- `Season`
- `Time of Day`
- `Urgency Level`
- `Nurse-to-Patient Ratio`
- `Specialist Availability`
- `Facility Size (Beds)`
- `hour`
- `day`
- `month`
- `day_of_week`

### Derived in the app
The app derives time-based fields automatically from the selected visit date and visit hour.

---

## 🗂 Project Structure

```bash
Emergency_Room_Wait_Time_Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── ER Wait Time Dataset.csv
│
├── models/
│   └── er_wait_model.pkl
│
├── notebooks/
│   ├── Hospital_ER_Wait_Time.ipynb
│   └── ER_Wait_Time_3.ipynb
│
└── .streamlit/
    └── config.toml
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
