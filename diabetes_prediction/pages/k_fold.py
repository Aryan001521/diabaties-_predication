import streamlit as st
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def run_k_fold_model():
    st.subheader("K-Fold Cross Validation (Logistic Regression)")

    if st.button("Run K-Fold Model"):
        # ✅ FIXED: Load Data from correct path
        df = pd.read_csv("diabetes.csv")
        
        st.write("Sample Data:")
        st.write(df.head())
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Initialize Stratified K-Fold
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold = 1
        plt.figure(figsize=(8, 6))  # Create a clean ROC figure
        
        for train_index, test_index in kfold.split(X, y):
            st.write(f"📁 Fold {fold}")
            fold += 1

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            st.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.write("🔍 Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("📝 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Fold {fold-1} AUC = {roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for K-Fold")
        plt.legend()
        st.pyplot(plt)
