import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st 
import io

def run_split_model():
    st.subheader("Split Validation")

    if st.button("Run Split Model"):
        df = pd.read_csv(r"C:\Users\aryan\OneDrive\Documents\phython project\diabetes_prediction\diabetes.csv")

        # Show data
        st.write("📋 Sample Data:")
        st.write(df.head())

        # Data Info
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text("📄 DataFrame Info:")
        st.text(s)

        st.write("📊 Summary Statistics:")
        st.write(df.describe())

        st.write("❓ Null Values:")
        st.write(df.isnull().sum())

        st.write(f"🧮 Shape of Dataset: {df.shape}")

        # Plot class balance
        st.write("📌 Diabetes Count Plot (Outcome 0 = No, 1 = Yes):")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Outcome', data=df, ax=ax1)
        ax1.set_title("Diabetes Count")
        st.pyplot(fig1)

        # Scatter Plot
        import plotly.express as px
        fig2 = px.scatter(
            df,
            x='Glucose',
            y='BMI',
            color='Outcome',
            hover_data=['Age', 'BloodPressure', 'Insulin'],
            title='Glucose vs BMI (Hover for Age, BP, Insulin)'
        )
        st.plotly_chart(fig2)

        # Features and Target
        x = df.drop("Outcome", axis=1)
        y = df['Outcome']

        # Split data
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Scale data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_train_scalar = scaler.fit_transform(x_train)
        x_test_scalar = scaler.transform(x_test)

        # Logistic Regression
        from sklearn.linear_model import LogisticRegression 
        model = LogisticRegression()
        model.fit(x_train_scalar, y_train)

        # Predictions and Evaluation
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
        y_pred = model.predict(x_test_scalar)

        st.write(f"✅ **Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.write("🔷 **Confusion Matrix:**")
        st.write(confusion_matrix(y_test, y_pred))

        st.write("📋 **Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # ROC Curve
        y_proba = model.predict_proba(x_test_scalar)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        st.write("📈 ROC Curve:")
        fig3, ax3 = plt.subplots()
        ax3.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
        ax3.plot([0, 1], [0, 1], 'k--')  # Random model diagonal
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curve")
        ax3.legend(loc="lower right")
        st.pyplot(fig3)
