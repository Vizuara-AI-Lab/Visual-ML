"""
Logistic Regression Interactive Guide

Comprehensive step-by-step instructions for building a logistic regression
classification model with context-aware explanations and dataset guidance.
"""

LOGISTIC_REGRESSION_GUIDE = {
    "model_type": "logistic_regression",
    "title": "Logistic Regression - Classify Data into Categories",
    # Introduction phase
    "introduction": {
        "title": "What is Logistic Regression?",
        "explanation": """**Logistic Regression** is a classification algorithm used to predict which category
something belongs to. Despite its name, it is used for classification — not regression.

**How it works:**
Logistic Regression uses the **sigmoid function** to convert any number into a probability between 0 and 1.
- If the probability is above a threshold (usually 0.5), it predicts class 1
- If below, it predicts class 0
- For multi-class problems, it extends this to multiple categories

**Key components:**
- **Features (X)**: Input variables (e.g., email word counts, patient symptoms)
- **Target (y)**: The category to predict (e.g., spam/not-spam, disease/healthy)
- **Sigmoid function**: Converts raw scores to probabilities
- **Decision boundary**: The line (or surface) that separates classes

**Real-World Examples:**
- 📧 **Spam Detection**: Is this email spam or not spam?
- 🏥 **Disease Diagnosis**: Does the patient have the disease?
- 💳 **Fraud Detection**: Is this transaction fraudulent?
- 🎓 **Student Outcome**: Will the student pass or fail?
- 🏦 **Loan Approval**: Should the loan be approved or denied?
- 🌸 **Species Classification**: Which species does this flower belong to?

**When to use Logistic Regression:**
✅ Your target is categorical (classes, labels, categories)
✅ You want probability estimates, not just predictions
✅ You need an interpretable model that shows feature importance
✅ Good baseline model for any classification problem

**When NOT to use it:**
❌ For predicting continuous numbers (use Linear Regression instead)
❌ When the relationship between features and target is highly non-linear (try Decision Tree)
❌ When you have very complex feature interactions (try Random Forest or MLP)
""",
        "visual_example": """
Example: Spam Detection
Feature (Word Counts)  →  Probability  →  Prediction
"free money now"       →  0.92         →  SPAM
"meeting at 3pm"       →  0.08         →  NOT SPAM
"claim your prize"     →  0.87         →  SPAM
"project update"       →  0.12         →  NOT SPAM

The sigmoid function converts scores to probabilities:
Score = -2.0 → Probability = 0.12 → NOT SPAM
Score = +2.5 → Probability = 0.92 → SPAM
""",
    },
    # Dataset preparation phase
    "dataset_phase": {
        "title": "Step 1: Prepare Your Dataset",
        "questions": [
            {
                "question": "Do you have a dataset ready to upload?",
                "options": [
                    {
                        "label": "Yes, I have my own dataset",
                        "action": "request_upload",
                        "next_message": """Great! Please make sure your dataset:
- Is in CSV format
- Has a **categorical target column** (the column you want to predict)
- The target should have a small number of categories (2-10 is ideal)
- Has numeric or categorical features

**⚠️ Important:** If your target column has numbers like 0, 1, 2, 3 — that's fine!
If it has text categories like "spam", "not_spam" — we'll encode it automatically.

**👉 Drag the 'Upload Dataset' node from the left panel onto the canvas.**""",
                    },
                    {
                        "label": "No, I want to use a sample dataset",
                        "action": "provide_sample",
                        "next_message": """Perfect! Here are great datasets for learning classification:

**Sample Datasets for Logistic Regression:**
1. 🚢 **Titanic** - Predict survival (survived/died) based on passenger features
2. 🏥 **Breast Cancer** - Classify tumors as malignant or benign
3. ❤️ **Heart Disease** - Predict presence of heart disease
4. 🌸 **Iris** - Classify iris flower species (3 classes)
5. 🐧 **Penguins** - Classify penguin species (3 classes)

Which would you like to try? (Or upload your own dataset)""",
                    },
                    {
                        "label": "I want to learn more about datasets first",
                        "action": "explain_dataset",
                        "next_message": """**Understanding Datasets for Classification:**

A classification dataset has:
- **Rows**: Individual examples (e.g., different emails, patients, passengers)
- **Feature columns**: The information used to make predictions
- **Target column**: The category you want to predict

**Example: Titanic Survival**
```
| Age | Gender | Class | Fare  | Survived |
|-----|--------|-------|-------|----------|
| 22  | Male   | 3     | 7.25  | 0 (No)   | ← One passenger
| 38  | Female | 1     | 71.28 | 1 (Yes)  |
| 26  | Female | 3     | 7.92  | 1 (Yes)  |
```

**Key differences from regression:**
- Target has **discrete categories** (not continuous numbers)
- We measure **accuracy** (not R² score)
- We use **Confusion Matrix** (not RMSE)

**Ready?** Drag the 'Upload Dataset' node from the left panel.""",
                    },
                ],
            }
        ],
    },
    # Main pipeline steps
    "pipeline_steps": [
        {
            "step": 1,
            "phase": "data_upload",
            "node_type": "upload_file",
            "title": "Upload Your Dataset",
            "description": "Load your CSV file containing features and target categories",
            "detailed_instructions": """**What to do:**
1. Look at the **left sidebar** under "Data Source" section
2. **Drag** the "Upload Dataset" node onto the canvas
3. **Click** on the node to configure it
4. **Browse** and select your CSV file
5. Click "Upload" and wait for processing

**What happens:**
- Your data is uploaded and validated
- I'll check for missing values and data types
- You'll see a preview of your dataset

**Tips:**
- Supported format: CSV files only
- Make sure your target column has categorical values
- If target has more than 10 unique values, consider Linear Regression instead""",
            "validation_checks": [
                "Dataset has been uploaded successfully",
                "Dataset contains a categorical target column",
                "Target column has 2-10 unique values",
            ],
            "common_errors": {
                "too_many_categories": "Target has too many unique values. Logistic Regression works best with 2-10 categories.",
                "continuous_target": "Target appears to be continuous numbers. Try Linear Regression instead.",
            },
        },
        {
            "step": 2,
            "phase": "data_analysis",
            "node_type": "data_analysis",
            "title": "Analyze Dataset Quality",
            "description": "Examine data quality and identify preprocessing needs",
            "automated": True,
            "detailed_instructions": """**I'm analyzing your dataset for:**

📊 **Data Quality Checks:**
- Missing values count and percentage
- Data types of each column
- Categorical vs numeric features
- Class balance (are categories equally represented?)

🎯 **Target Variable Analysis:**
- Number of unique categories
- Class distribution (balanced or imbalanced?)
- If severely imbalanced, I'll suggest strategies

📈 **Feature Analysis:**
- Identifying categorical columns that need encoding
- Checking for high-cardinality columns
- Detecting potential issues""",
        },
        {
            "step": 3,
            "phase": "handle_missing_values",
            "node_type": "missing_value_handler",
            "title": "Handle Missing Values",
            "description": "Clean missing/null values in your dataset",
            "skip_condition": "no_missing_values",
            "detailed_instructions": """**Why this matters for classification:**
Missing values can bias predictions toward certain classes.

**What to do:**
1. Drag the "Handle Missing Values" node from the left panel
2. Connect it to your uploaded dataset node
3. Choose a strategy:
   - **Mean/Median**: For numeric columns
   - **Mode**: For categorical columns (fills with most common category)
   - **Drop Rows**: If few rows have missing values (<5%)

**My recommendation for classification:**
Mode is usually best for classification datasets because it preserves the
class distribution. For numeric features, median is more robust than mean.

**Tip:** Don't drop too many rows — classification models need enough
examples of each class to learn effectively.""",
        },
        {
            "step": 4,
            "phase": "encoding",
            "node_type": "encoding",
            "title": "Encode Categorical Features",
            "description": "Convert text/category columns to numbers",
            "skip_condition": "no_categorical_columns",
            "detailed_instructions": """**Why this matters:**
Logistic Regression needs numbers, not text. We must convert categories to numbers.

**Important:** This step encodes **feature** columns. If your **target** column
is text (like "spam"/"not_spam"), the model will handle that automatically.

**What to do:**
1. Drag the "Encoding" node from the preprocessing section
2. Connect it after the missing value handler
3. Choose encoding method per column:
   - **Label Encoding**: For ordinal data (Small=0, Medium=1, Large=2)
   - **One-Hot Encoding**: For nominal data (Red, Blue, Green → 3 separate columns)

**My recommendation:**
Use One-Hot Encoding for most categorical features. It prevents the model
from assuming an ordering relationship between categories.

**⚠️ Watch out:** One-Hot Encoding on high-cardinality columns (50+ categories)
creates too many features. Use Label Encoding for those.""",
        },
        {
            "step": 5,
            "phase": "train_test_split",
            "node_type": "split",
            "title": "Split Data into Train & Test Sets",
            "description": "Divide data for training and evaluation",
            "detailed_instructions": """**Why this matters for classification:**
We need separate data to train and evaluate the model fairly.

**What to do:**
1. Drag the "Train-Test Split" node
2. Connect it after encoding
3. **Select your target column** — the category you want to predict
4. Configure split ratio:
   - **80/20**: Standard split (recommended)
   - **70/30**: If you have less data

**Important for classification:**
- Enable **Stratified Split** — this ensures each class is proportionally
  represented in both train and test sets
- Without stratification, a rare class might end up entirely in one set

**Example:** If your dataset has 90% "not spam" and 10% "spam", stratified
split ensures both train and test have roughly 90/10 ratio.

**Tip:** Choose the correct target column — this is the column you want to predict.""",
        },
        {
            "step": 6,
            "phase": "train_model",
            "node_type": "logistic_regression",
            "title": "Train Logistic Regression Model",
            "description": "Build and train your classification model",
            "detailed_instructions": """**This is the exciting part!** 🎉

**What happens:**
The algorithm learns the best decision boundary by:
1. Computing weighted sums of features
2. Passing through the sigmoid function to get probabilities
3. Adjusting weights to minimize classification errors
4. Iterating until convergence

**What to do:**
1. Drag the "Logistic Regression" node from ML Algorithms section
2. Connect it to the train-test split node

**Configuration tips:**
- **C (Regularization)**: Controls model complexity
  - Higher C = more complex model (fits training data closely)
  - Lower C = simpler model (better generalization)
  - Default: 1.0 (good starting point)
- **penalty**: Type of regularization
  - 'l2' (default) — shrinks all coefficients evenly
  - 'l1' — can zero out unimportant features (feature selection)
- **solver**: Optimization algorithm
  - 'lbfgs' (default) — works for all problems
  - 'liblinear' — good for small datasets
  - 'saga' — good for large datasets
- **max_iter**: Maximum iterations (increase to 1000 if not converging)

**You'll see:**
- Model coefficients (which features influence each class)
- Training accuracy
- Convergence status""",
            "success_message": """🎉 **Model Training Complete!**

Your logistic regression model has learned to classify your data!

**Next:** Let's evaluate how well it performs with a Confusion Matrix.""",
        },
        {
            "step": 7,
            "phase": "evaluate",
            "node_type": "confusion_matrix",
            "title": "Evaluate with Confusion Matrix",
            "description": "Visualize prediction accuracy per class",
            "detailed_instructions": """**Understanding the Confusion Matrix:**

The confusion matrix shows exactly what the model got right and wrong:

```
                  Predicted
              | Positive | Negative |
Actual  Pos   |    TP    |    FN    |
        Neg   |    FP    |    TN    |
```

**Key terms:**
- **True Positive (TP)**: Correctly predicted positive (e.g., correctly caught spam)
- **True Negative (TN)**: Correctly predicted negative (e.g., correctly passed real email)
- **False Positive (FP)**: Wrongly predicted positive (e.g., real email marked as spam)
- **False Negative (FN)**: Wrongly predicted negative (e.g., spam that got through)

**Key metrics from the confusion matrix:**
- **Accuracy** = (TP + TN) / Total — overall correct predictions
- **Precision** = TP / (TP + FP) — "of all predicted positive, how many truly are?"
- **Recall** = TP / (TP + FN) — "of all actual positives, how many did we catch?"
- **F1 Score** = harmonic mean of precision and recall

**⚠️ Important:** Accuracy alone can be misleading!
If 95% of emails are not spam, a model that always predicts "not spam" gets 95% accuracy
but catches zero spam. Always check precision and recall too.

**What to do:**
1. Drag the "Confusion Matrix" node from the Results section
2. Connect it to your trained model
3. View the results

**Interpreting results:** I'll help you understand if your model is good enough!""",
            "success_message": """📊 **Evaluation Complete!**

{metrics_results}

**Analysis:** {performance_analysis}

**Tips for improvement:**
- If precision is low: the model makes too many false positive errors
- If recall is low: the model misses too many actual positives
- Try adjusting the C parameter or adding more features""",
        },
    ],
    # Completion messages
    "completion": {
        "success": """🎉 **Congratulations!** You've successfully built a Logistic Regression classifier!

**What you accomplished:**
✅ Uploaded and analyzed dataset
✅ Preprocessed data (handling missing values, encoding)
✅ Split data with stratification
✅ Trained a logistic regression model
✅ Evaluated with confusion matrix

**What you can do next:**
1. **Adjust the C parameter**: Try C=0.1 (simpler) or C=10 (more complex)
2. **Try different features**: Remove or add features to improve accuracy
3. **Check class balance**: If one class dominates, try stratified sampling
4. **Compare with other models**: Try Decision Tree or Random Forest
5. **Look at feature coefficients**: See which features matter most

**Want to learn more?**
- How the sigmoid function works
- Regularization and overfitting
- Multi-class classification strategies
- ROC curves and AUC score""",
        "poor_performance": """**Model Training Complete** - Let's improve! 📊

Your accuracy suggests the model could be better. Here's what we can try:

**Common improvements for classification:**
1. **Handle class imbalance**: If one class has many more samples, try:
   - Oversampling the minority class
   - Undersampling the majority class
   - Adjusting class weights
2. **Feature engineering**: Create new features from existing ones
3. **Try different C values**: C=0.01 to C=100
4. **Different algorithm**: Try Decision Tree or Random Forest for non-linear patterns

**Data quality checks:**
- Is your target truly categorical?
- Do features have predictive power for the target?
- Is there enough data per class (ideally 50+ samples)?""",
    },
    "estimated_time": "5-10 minutes",
    "prerequisites": [
        "Dataset in CSV format",
        "Categorical target column (2-10 classes ideal)",
        "Minimum 50 rows of data recommended",
    ],
}
