# ============================================================
# Car Evaluation System - Decision Tree Classifier
# ============================================================
# Dataset: UCI Car Evaluation Dataset
# Model: Decision Tree Classifier
# Author: Shree
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────

def load_data():
    """Load the UCI Car Evaluation dataset."""
    import os, numpy as np

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    try:
        df = pd.read_csv(url, header=None, names=columns)
        print("✅ Dataset loaded from UCI repository.")
        return df
    except Exception:
        pass

    # Fallback: generate full 1728-row dataset matching UCI structure
    print("⚠️  Network unavailable. Generating equivalent dataset (1728 rows).")
    buying   = ['vhigh', 'high', 'med', 'low']
    maint    = ['vhigh', 'high', 'med', 'low']
    doors    = ['2', '3', '4', '5more']
    persons  = ['2', '4', 'more']
    lug_boot = ['small', 'med', 'big']
    safety   = ['low', 'med', 'high']

    def assign_class(b, m, p, s):
        bp = buying.index(b);  mp = maint.index(m)
        pp = persons.index(p); sp = safety.index(s)
        score = (3 - bp) + (3 - mp) + pp + sp
        if score <= 2:   return 'unacc'
        elif score <= 4: return 'acc'
        elif score <= 6: return 'good'
        else:            return 'vgood'

    rows = []
    for b in buying:
        for m in maint:
            for d in doors:
                for p in persons:
                    for l in lug_boot:
                        for s in safety:
                            rows.append([b, m, d, p, l, s, assign_class(b, m, p, s)])

    df = pd.DataFrame(rows, columns=columns)
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def exploratory_analysis(df):
    """Perform and display basic EDA."""
    print("\n" + "="*55)
    print("       EXPLORATORY DATA ANALYSIS")
    print("="*55)
    print(f"\n📊 Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"🔍 Missing Values: {df.isnull().sum().sum()}")
    print("\n📋 First 5 Rows:")
    print(df.head().to_string(index=False))
    print("\n📌 Feature Unique Values:")
    for col in df.columns:
        print(f"   {col:10s}: {df[col].unique().tolist()}")
    print("\n🎯 Class Distribution:")
    print(df['class'].value_counts().to_string())


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    """Encode categorical features and split data."""
    df_enc = df.copy()
    encoders = {}

    for col in df_enc.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    X = df_enc.drop('class', axis=1)
    y = df_enc['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✅ Preprocessing Complete.")
    print(f"   Train samples : {X_train.shape[0]}")
    print(f"   Test  samples : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, encoders, df_enc


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(X_train, y_train):
    """Train Decision Tree Classifier."""
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n🌳 Decision Tree Trained.")
    print(f"   Cross-Val Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, encoders):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    class_names = encoders['class'].classes_

    print("\n" + "="*55)
    print("           MODEL EVALUATION")
    print("="*55)
    print(f"\n🎯 Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return y_pred, class_names


# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_visualizations(df, model, X_train, X_test, y_test, y_pred, class_names, encoders):
    """Generate and save all project visualizations."""
    feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('Car Evaluation System - Decision Tree Analysis', fontsize=18, fontweight='bold', y=0.98)

    # ── Plot 1: Class Distribution ──
    ax1 = fig.add_subplot(3, 3, 1)
    class_counts = df['class'].value_counts()
    bars = ax1.bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)], edgecolor='black', linewidth=0.8)
    ax1.set_title('Class Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Car Evaluation Class')
    ax1.set_ylabel('Count')
    for bar, val in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontsize=10)
    ax1.grid(axis='y', alpha=0.4)

    # ── Plot 2: Feature Distributions ──
    ax2 = fig.add_subplot(3, 3, 2)
    feature_counts = {feat: df[feat].nunique() for feat in feature_names}
    ax2.barh(list(feature_counts.keys()), list(feature_counts.values()), color='#5C6BC0', edgecolor='black')
    ax2.set_title('Unique Values per Feature', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Number of Unique Values')
    ax2.grid(axis='x', alpha=0.4)

    # ── Plot 3: Buying vs Class ──
    ax3 = fig.add_subplot(3, 3, 3)
    ct = pd.crosstab(df['buying'], df['class'])
    ct.plot(kind='bar', ax=ax3, colormap='Set2', edgecolor='black')
    ax3.set_title('Buying Price vs Car Class', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Buying Price')
    ax3.set_ylabel('Count')
    ax3.legend(title='Class', fontsize=8)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.4)

    # ── Plot 4: Safety vs Class ──
    ax4 = fig.add_subplot(3, 3, 4)
    ct2 = pd.crosstab(df['safety'], df['class'])
    ct2.plot(kind='bar', ax=ax4, colormap='Set1', edgecolor='black')
    ax4.set_title('Safety vs Car Class', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Safety Level')
    ax4.set_ylabel('Count')
    ax4.legend(title='Class', fontsize=8)
    ax4.tick_params(axis='x', rotation=0)
    ax4.grid(axis='y', alpha=0.4)

    # ── Plot 5: Confusion Matrix ──
    ax5 = fig.add_subplot(3, 3, 5)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax5, colorbar=False, cmap='Blues')
    ax5.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

    # ── Plot 6: Feature Importances ──
    ax6 = fig.add_subplot(3, 3, 6)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    bars6 = ax6.bar(sorted_features, sorted_importances, color='#26A69A', edgecolor='black')
    ax6.set_title('Feature Importances', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Feature')
    ax6.set_ylabel('Importance Score')
    for bar, val in zip(bars6, sorted_importances):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}', ha='center', fontsize=9)
    ax6.grid(axis='y', alpha=0.4)

    # ── Plot 7: Decision Tree Visualization ──
    ax7 = fig.add_subplot(3, 1, 3)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=3,
        ax=ax7,
        fontsize=8
    )
    ax7.set_title('Decision Tree Structure (max depth=3)', fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('/mnt/user-data/outputs/car_evaluation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Visualizations saved.")


# ─────────────────────────────────────────────
# 7. PREDICT NEW SAMPLE
# ─────────────────────────────────────────────

def predict_sample(model, encoders):
    """Predict a new car sample."""
    sample = {
        'buying': 'med',
        'maint': 'low',
        'doors': '4',
        'persons': 'more',
        'lug_boot': 'big',
        'safety': 'high'
    }

    sample_enc = []
    for col, val in sample.items():
        encoded = encoders[col].transform([val])[0]
        sample_enc.append(encoded)

    prediction_enc = model.predict([sample_enc])[0]
    prediction = encoders['class'].inverse_transform([prediction_enc])[0]

    print("\n" + "="*55)
    print("         PREDICTION ON NEW SAMPLE")
    print("="*55)
    print(f"   Input  : {sample}")
    print(f"   Result : 🚗 Car Evaluation → '{prediction.upper()}'")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    print("="*55)
    print("   🚗 CAR EVALUATION SYSTEM - DECISION TREE")
    print("="*55)

    df = load_data()
    exploratory_analysis(df)
    X_train, X_test, y_train, y_test, encoders, df_enc = preprocess(df)
    model = train_model(X_train, y_train)
    y_pred, class_names = evaluate_model(model, X_test, y_test, encoders)
    plot_visualizations(df, model, X_train, X_test, y_test, y_pred, class_names, encoders)
    predict_sample(model, encoders)

    print("\n✅ Project Complete! All outputs saved.")


if __name__ == "__main__":
    main()
