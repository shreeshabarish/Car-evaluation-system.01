# 🚗 Car Evaluation System — Decision Tree Classifier

A machine learning project that predicts the **acceptability of a car** based on its features using a **Decision Tree Classifier**.

---

## 📌 Project Overview

This project uses the **UCI Car Evaluation Dataset** to classify cars into four categories:

| Class | Meaning |
|-------|---------|
| `unacc` | Unacceptable |
| `acc` | Acceptable |
| `good` | Good |
| `vgood` | Very Good |

---

## 📂 Project Structure

```
car-evaluation-ml/
│
├── car_evaluation.py            # Main Python script
├── Car_Evaluation_System.ipynb  # Jupyter Notebook (step-by-step)
├── README.md                    # Project documentation
└── car_evaluation_analysis.png  # Generated visualizations
```

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
- **Samples:** 1728
- **Features:** 6 categorical attributes
- **Target:** Car acceptability class

### Features

| Feature | Description | Values |
|---------|-------------|--------|
| `buying` | Buying price | vhigh, high, med, low |
| `maint` | Maintenance cost | vhigh, high, med, low |
| `doors` | Number of doors | 2, 3, 4, 5more |
| `persons` | Passenger capacity | 2, 4, more |
| `lug_boot` | Luggage boot size | small, med, big |
| `safety` | Safety rating | low, med, high |

---

## 🤖 Model

**Algorithm:** Decision Tree Classifier (scikit-learn)

```python
DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Cross-Validation Accuracy (5-fold) | ~69% |
| Test Accuracy | ~67% |
| Best Class F1-Score | `vgood` — 0.84 |

---

## 🔍 Key Insights

- **`safety`** and **`persons`** are the most influential features for car acceptability
- Cars with **high safety** and capacity for **more persons** tend to be rated `good` or `vgood`
- **Buying price** and **maintenance cost** heavily influence `unacc` classification

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Python Script

```bash
python car_evaluation.py
```

### Run the Jupyter Notebook

```bash
jupyter notebook Car_Evaluation_System.ipynb
```

---

## 📊 Visualizations

The project generates the following plots:

1. **Class Distribution** — Bar chart of car classes
2. **Unique Values per Feature** — Feature cardinality
3. **Buying Price vs Class** — Cross-tabulation
4. **Safety vs Class** — Cross-tabulation
5. **Confusion Matrix** — Model prediction accuracy
6. **Feature Importances** — Which features matter most
7. **Decision Tree Structure** — Visual tree (max depth 3)

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 👤 Author

**Shree**  
AI/ML Student

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
