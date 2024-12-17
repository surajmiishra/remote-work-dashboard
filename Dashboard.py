import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_csv('Impact_of_Remote_Work_on_Mental_Health.csv')

# Streamlit app title
st.title("Advanced Interactive Dashboard for Remote Work & Burnout Analysis")

# Sidebar Filters
st.sidebar.header("Filter Data")
selected_stress_levels = st.sidebar.multiselect("Select Stress Levels", options=df['Stress_Level'].unique(), default=df['Stress_Level'].unique())
selected_work_location = st.sidebar.multiselect("Select Work Locations", options=df['Work_Location'].unique(), default=df['Work_Location'].unique())
selected_columns = st.sidebar.multiselect("Select Columns for Analysis", options=df.columns, default=['Stress_Level', 'Work_Location', 'Work_Life_Balance_Rating', 'Hours_Worked_Per_Week'])

# Filter data based on selections
filtered_df = df[(df['Stress_Level'].isin(selected_stress_levels)) & (df['Work_Location'].isin(selected_work_location))]

# --- Section 1: Stress Levels by Work Location ---
st.subheader("1. Stress Levels by Work Location")
fig1 = px.histogram(filtered_df, x='Stress_Level', color='Work_Location', barmode='group', title="Stress Levels by Work Location")
st.plotly_chart(fig1)

# --- Section 2: Work-Life Balance and Work Hours ---
st.subheader("2. Work-Life Balance vs. Work Hours")
fig2 = px.box(filtered_df, x='Work_Life_Balance_Rating', y='Hours_Worked_Per_Week', color='Work_Location', title="Impact of Work Hours on Work-Life Balance")
st.plotly_chart(fig2)

# --- Section 3: Stress Levels and Mental Health Resource Access ---
st.subheader("3. Stress Levels and Mental Health Resource Access")
fig3 = px.histogram(filtered_df, x='Stress_Level', color='Access_to_Mental_Health_Resources', barmode='group', title="Stress Levels and Mental Health Resource Access")
st.plotly_chart(fig3)

# --- Section 4: Correlation Heatmap ---
st.subheader("4. Correlation Heatmap")
if len(selected_columns) > 1:
    numeric_features = filtered_df[selected_columns].select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_features.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
else:
    st.write("Please select at least two numeric columns for correlation analysis.")

# --- Section 5: Productivity Change by Work Location ---
st.subheader("5. Productivity Change by Work Location")
fig5 = px.histogram(filtered_df, x='Productivity_Change', color='Work_Location', barmode='group', title="Productivity Change by Work Location")
st.plotly_chart(fig5)

# --- Section 6: Social Isolation and Sleep Quality ---
st.subheader("6. Social Isolation Rating vs. Sleep Quality")
fig6 = px.box(filtered_df, x='Sleep_Quality', y='Social_Isolation_Rating', title="Social Isolation Rating and Sleep Quality")
st.plotly_chart(fig6)

# --- Section 7: Machine Learning - Feature Importance and Prediction ---
st.subheader("7. Machine Learning Analysis")
ml_columns = st.multiselect("Select Features for Model", options=df.columns.drop(['Stress_Level']), default=['Hours_Worked_Per_Week', 'Work_Life_Balance_Rating'])
if ml_columns:
    X = df[ml_columns].dropna()
    y = df.loc[X.index, 'Stress_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importance
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': ml_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    fig7 = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig7)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig)
else:
    st.write("Please select features for the model.")

# --- Section 8: Summary Metrics ---
st.subheader("8. Summary Statistics")
st.write(filtered_df.describe())

# --- Section 9: Interactive Data Table ---
st.subheader("9. Explore Filtered Data")
st.dataframe(filtered_df)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("Advanced Interactive Dashboard for Remote Work Analysis\n\nBuilt with Streamlit & Plotly")
