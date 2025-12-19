import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(page_title="Wine Classification", layout="wide")

st.title(" Wine Classification Using Naive Bayes")
st.markdown("### Multi-Class Classification Problem (3 Wine Types)")

st.info(" **What is this?** This app classifies wines into 3 categories based on 13 chemical properties using a Naive Bayes classifier.")

# ============================================
# STEP 1: LOAD THE DATASET
# ============================================
st.header("Step 1: Load the Wine Dataset ")

# Load the wine dataset from scikit-learn
wine = datasets.load_wine()

st.success(" Dataset loaded successfully!")

# Show dataset information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", wine.data.shape[0])
with col2:
    st.metric("Number of Features", wine.data.shape[1])
with col3:
    st.metric("Number of Classes", len(wine.target_names))

# ============================================
# STEP 2: EXPLORE THE FEATURES
# ============================================
st.header("Step 2: Understand the Features ")

st.markdown("What are the feature names?**")

# Print the names of the 13 features
st.write("**Features (13 chemical properties):**")
st.write(wine.feature_names)

# Show features in a nice format
st.markdown("#### All 13 Features:")
features_df = pd.DataFrame(wine.feature_names, columns=["Feature Name"])
features_df.index = features_df.index + 1
st.dataframe(features_df, use_container_width=True)

st.markdown("""
**What are features?** Features are the input variables (measurements) we use to predict the wine type.
For example: alcohol content, color intensity, pH level, etc.
""")

# ============================================
# STEP 3: EXPLORE THE LABELS
# ============================================
st.header("Step 3: Understand the Labels ")

st.markdown("**What are the label names?**")

# Print the label type of wine (Class_0, Class_1, Class_2)
st.write("**Labels (Wine Classes):**")
st.write(wine.target_names)

# Show class distribution
st.markdown("#### Class Distribution:")
target_counts = pd.Series(wine.target).value_counts().sort_index()
target_df = pd.DataFrame({
    'Class': wine.target_names,
    'Count': target_counts.values
})
st.dataframe(target_df, use_container_width=True)

st.markdown("""
**What are labels?** Labels are the output we want to predict (the wine type).
- Class_0, Class_1, Class_2 represent 3 different wine varieties.
""")

# Show sample data
st.markdown("#### Sample Data (First 5 Rows):")
sample_df = pd.DataFrame(wine.data[:5], columns=wine.feature_names)
sample_df['Target (Wine Class)'] = wine.target[:5]
sample_df['Class Name'] = [wine.target_names[i] for i in wine.target[:5]]
st.dataframe(sample_df, use_container_width=True)

# ============================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING
# ============================================
st.header("Step 4: Split Data into Training and Testing Sets ")

st.markdown("**Split the dataset (70% training, 30% testing)**")

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    wine.data,           # Features (input)
    wine.target,         # Labels (output)
    test_size=0.3,       # 30% of data for testing
    random_state=109     # Fixed seed for reproducibility
)

st.success("Data split completed!")

# Show the split information
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Training Set (70%)")
    st.metric("Training Samples", len(X_train))
    st.write(f"- Features shape: {X_train.shape}")
    st.write(f"- Labels shape: {y_train.shape}")
    
with col2:
    st.markdown("#### Testing Set (30%)")
    st.metric("Testing Samples", len(X_test))
    st.write(f"- Features shape: {X_test.shape}")
    st.write(f"- Labels shape: {y_test.shape}")

st.markdown("""
**Why split the data?**
- **Training Set**: Used to teach the model (like studying for an exam)
- **Testing Set**: Used to evaluate the model on unseen data (like taking the actual exam)
- This helps us know if the model can generalize to new data
""")

# ============================================
# STEP 5: CREATE AND TRAIN THE MODEL
# ============================================
st.header("Step 5: Create and Train the Naive Bayes Model ")

st.markdown("**Create, train, and make predictions**")

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
st.markdown("#### 5.1: Create the Classifier")
gnb = GaussianNB()
st.code("gnb = GaussianNB()")
st.success("Gaussian Naive Bayes classifier created!")

st.markdown("""
**What is Gaussian Naive Bayes?**
- A machine learning algorithm that assumes features follow a normal (Gaussian) distribution
- "Naive" because it assumes features are independent of each other
- Works well for classification problems
""")

# Train the model using the training sets
st.markdown("#### 5.2: Train the Model")
gnb.fit(X_train, y_train)
st.code("gnb.fit(X_train, y_train)")
st.success("Model trained successfully!")

st.markdown("""
**What is training?**
- The model learns patterns from the training data
- It calculates probabilities for each feature-class combination
- This process is also called "fitting" the model
""")

# Predict the response for test dataset
st.markdown("#### 5.3: Make Predictions")
y_pred = gnb.predict(X_test)
st.code("y_pred = gnb.predict(X_test)")
st.success("Predictions completed!")

st.markdown("""
**What is prediction?**
- The model uses what it learned to classify new wine samples
- It predicts the wine class for each sample in the test set
""")

# Show some predictions vs actual
st.markdown("#### Sample Predictions (First 10):")
predictions_df = pd.DataFrame({
    'Sample #': range(1, 11),
    'Actual Class': [wine.target_names[i] for i in y_test[:10]],
    'Predicted Class': [wine.target_names[i] for i in y_pred[:10]],
    'Correct?': ['yes' if y_test[i] == y_pred[i] else 'no' for i in range(10)]
})
st.dataframe(predictions_df, use_container_width=True)

# ============================================
# STEP 6: EVALUATE THE MODEL
# ============================================
st.header("Step 6: Evaluate Model Performance ")

st.markdown("**Calculate the accuracy**")

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
accuracy = metrics.accuracy_score(y_test, y_pred)

st.code("accuracy = metrics.accuracy_score(y_test, y_pred)")

# Display accuracy prominently
st.markdown("###  Model Accuracy")
st.metric("Accuracy Score", f"{accuracy:.4f} ({accuracy*100:.2f}%)")

# Progress bar visualization
st.progress(accuracy)

# Interpretation
if accuracy >= 0.95:
    st.success(" Excellent! The model performs very well!")
elif accuracy >= 0.85:
    st.success(" Good! The model performs well.")
elif accuracy >= 0.70:
    st.warning(" Fair. The model could be improved.")
else:
    st.error(" Poor. The model needs significant improvement.")

st.markdown("""
**What is accuracy?**
- Accuracy = (Correct Predictions) / (Total Predictions)
- It tells us how often the model is correct
- Higher accuracy = better model performance
""")

# Additional metrics
st.markdown("###  Detailed Metrics")

total_predictions = len(y_test)
correct_predictions = (y_test == y_pred).sum()
incorrect_predictions = total_predictions - correct_predictions

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Predictions", total_predictions)
with col2:
    st.metric("Correct Predictions", correct_predictions)
with col3:
    st.metric("Incorrect Predictions", incorrect_predictions)

# Confusion Matrix
st.markdown("###  Confusion Matrix")
st.markdown("Shows how many predictions were correct/incorrect for each class")

cm = metrics.confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f'Actual {name}' for name in wine.target_names],
    columns=[f'Predicted {name}' for name in wine.target_names]
)
st.dataframe(cm_df, use_container_width=True)

st.markdown("""
**How to read the confusion matrix:**
- Diagonal values (top-left to bottom-right): **Correct predictions**
- Off-diagonal values: **Incorrect predictions**
- Example: If cell shows "5" at row "Actual Class_0" and column "Predicted Class_0", 
  it means 5 Class_0 wines were correctly predicted
""")

# Classification Report
st.markdown("###  Classification Report")
report = metrics.classification_report(y_test, y_pred, target_names=wine.target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)

st.markdown("""
**Metrics Explained:**
- **Precision**: When the model predicts a class, how often is it correct?
- **Recall**: Of all actual samples of a class, how many did the model find?
- **F1-Score**: Balanced measure of precision and recall
- **Support**: Number of samples of each class in the test set
""")

# ============================================
# SUMMARY
# ============================================
st.header("Summary")

st.success(f"""
###  Project Complete!

**What we did:**
1. Loaded the Wine dataset (178 samples, 13 features, 3 classes)
2. Explored features (chemical properties) and labels (wine types)
3. Split data into training (70%) and testing (30%) sets
4. Created a Gaussian Naive Bayes classifier
5. Trained the model on training data
6. Made predictions on testing data
7. Evaluated model accuracy: **{accuracy*100:.2f}%**

**Key Takeaway:**
The Naive Bayes classifier successfully learned to classify wines into 3 categories 
based on their chemical properties with an accuracy of {accuracy*100:.2f}%!
""")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Wine Classification Project</strong></p>
    <p>Using Scikit-learn's Wine Dataset & Gaussian Naive Bayes Classifier</p>
</div>
""", unsafe_allow_html=True)
