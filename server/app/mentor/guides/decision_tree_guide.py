"""
Decision Tree Interactive Guide

Comprehensive step-by-step instructions for building a decision tree
model with context-aware explanations and dataset guidance.
"""

DECISION_TREE_GUIDE = {
    "model_type": "decision_tree",
    "title": "Decision Tree - Flowchart-Style Predictions",
    # Introduction phase
    "introduction": {
        "title": "What is a Decision Tree?",
        "explanation": """**Decision Tree** is a model that makes predictions by asking a series of
yes/no questions — just like a flowchart. It splits the data step by step until it
reaches a final prediction.

**How it works:**
A Decision Tree learns the best questions to ask about your data:
1. Start at the top (root) with all data
2. Ask the most informative question (e.g., "Is income > $50k?")
3. Split data into two branches based on the answer
4. Repeat until each branch has a clear prediction

**Key components:**
- **Root Node**: The first question — the most important split
- **Internal Nodes**: Follow-up questions that refine the prediction
- **Leaf Nodes**: Final predictions (the answers)
- **Branches**: The paths connecting questions to answers
- **Depth**: How many levels of questions the tree asks

**Real-World Examples:**
- 🏦 **Loan Approval**: Income > $50k? → Credit score > 700? → Approved!
- 🏥 **Medical Diagnosis**: Fever? → Cough? → Flu vs Cold
- 🎮 **Game Recommendation**: Likes action? → Multiplayer? → Suggest game
- 🏠 **House Pricing**: Location? → Size? → Predicted price
- 📧 **Email Filtering**: Contains "urgent"? → From known sender? → Priority

**Decision Tree can do BOTH:**
✅ **Classification**: Predict categories (spam/not-spam, approve/deny)
✅ **Regression**: Predict numbers (house price, temperature, score)

**When to use Decision Trees:**
✅ You want an easily interpretable model (can visualize the tree)
✅ Your data has non-linear relationships
✅ You have mixed data types (numeric + categorical)
✅ You want to see which features matter most

**When NOT to use them:**
❌ You need very stable predictions (small data changes → different tree)
❌ You have very high-dimensional data (risk of overfitting)
❌ You need the best possible accuracy (try Random Forest instead)""",
        "visual_example": """
Example: Loan Approval Decision Tree

                    [Income > $50k?]
                   /                \\
                 Yes                 No
                /                     \\
      [Credit > 700?]          [Has Cosigner?]
       /          \\              /          \\
     Yes          No           Yes          No
      |            |            |            |
  APPROVED    REVIEW       APPROVED      DENIED

Each path from top to bottom is a rule the model learned!
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
- Has a clear **target column** (what you want to predict)
- For **classification**: target should be categorical (labels, classes)
- For **regression**: target should be numeric (prices, scores, quantities)

**Decision Trees are versatile!** They handle both types naturally:
- Categories like "approved"/"denied" → Classification Tree
- Numbers like house prices → Regression Tree

**👉 Drag the 'Upload Dataset' node from the left panel onto the canvas.**""",
                    },
                    {
                        "label": "No, I want to use a sample dataset",
                        "action": "provide_sample",
                        "next_message": """Perfect! Here are great datasets for learning Decision Trees:

**For Classification (predicting categories):**
1. 🚢 **Titanic** - Predict survival based on passenger info
2. 🏥 **Breast Cancer** - Classify tumors as malignant or benign
3. 🌸 **Iris** - Classify flower species (great for visualizing trees!)

**For Regression (predicting numbers):**
4. 🏠 **Housing** - Predict house prices
5. 🚗 **Auto MPG** - Predict fuel efficiency
6. 💰 **Tips** - Predict restaurant tip amounts

Which would you like to try? (Or upload your own dataset)""",
                    },
                    {
                        "label": "I want to learn more about datasets first",
                        "action": "explain_dataset",
                        "next_message": """**Understanding Datasets for Decision Trees:**

A dataset is a table where:
- **Rows**: Individual examples (patients, houses, emails)
- **Feature columns**: Information used to make decisions
- **Target column**: What you want to predict

**Example: Loan Approval**
```
| Income | Credit | Age | Employed | Approved |
|--------|--------|-----|----------|----------|
| 55000  | 720    | 35  | Yes      | Yes      | ← One applicant
| 32000  | 580    | 22  | No       | No       |
| 68000  | 690    | 45  | Yes      | Yes      |
```

**The tree will learn rules like:**
- IF Income > $50k AND Credit > 700 → Approved
- IF Income < $30k AND not Employed → Denied

**Key advantage:** Decision Trees handle BOTH numeric (Income, Credit)
and categorical (Employed: Yes/No) features naturally!

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
            "description": "Load your CSV file containing features and a target column",
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

**Tips for Decision Trees:**
- Decision Trees handle missing data better than most algorithms
- They work with both numeric and categorical features
- No need to worry about feature scaling (trees don't care about scale!)""",
            "validation_checks": [
                "Dataset has been uploaded successfully",
                "Dataset contains a clear target column",
                "Dataset has at least 2 feature columns",
            ],
            "common_errors": {
                "too_few_rows": "Very small datasets may lead to overfitting. Aim for 50+ rows.",
                "single_value_target": "Target column has only one unique value. Check your data.",
            },
        },
        {
            "step": 2,
            "phase": "data_analysis",
            "node_type": "data_analysis",
            "title": "Analyze Dataset Quality",
            "description": "Examine data quality and understand feature types",
            "automated": True,
            "detailed_instructions": """**I'm analyzing your dataset for:**

📊 **Data Quality Checks:**
- Missing values count and percentage
- Data types of each column
- Categorical vs numeric features

🎯 **Target Variable Analysis:**
- Is it categorical (classification) or numeric (regression)?
- Number of unique values
- Distribution of values

🌳 **Decision Tree Specific Checks:**
- Features with high cardinality (many unique values)
- Features that might cause overfitting
- Recommended tree depth based on dataset size""",
        },
        {
            "step": 3,
            "phase": "handle_missing_values",
            "node_type": "missing_value_handler",
            "title": "Handle Missing Values",
            "description": "Clean missing/null values in your dataset",
            "skip_condition": "no_missing_values",
            "detailed_instructions": """**Why this matters:**
While Decision Trees can sometimes handle missing values, it's best practice
to clean them for consistent results.

**What to do:**
1. Drag the "Handle Missing Values" node from the left panel
2. Connect it to your uploaded dataset node
3. Choose a strategy:
   - **Mean**: For numeric columns (fills with average)
   - **Median**: More robust for numeric columns with outliers
   - **Mode**: For categorical columns (fills with most common value)
   - **Drop Rows**: If only a few rows have missing values (<5%)

**Decision Tree tip:**
Since trees split on individual features, filling missing values with
median (numeric) or mode (categorical) works well. The tree can still
learn the correct splits even with imputed values.

**⚠️ Avoid** dropping too many rows — trees need enough examples to
learn good splits at each level.""",
        },
        {
            "step": 4,
            "phase": "encoding",
            "node_type": "encoding",
            "title": "Encode Categorical Features",
            "description": "Convert text/category columns to numbers",
            "skip_condition": "no_categorical_columns",
            "detailed_instructions": """**Why this matters:**
Although Decision Trees conceptually handle categories, the implementation
needs numeric input. We must convert text labels to numbers.

**What to do:**
1. Drag the "Encoding" node from the preprocessing section
2. Connect it after the missing value handler
3. Choose encoding method per column:
   - **Label Encoding**: Converts categories to numbers (Red=0, Blue=1, Green=2)
   - **One-Hot Encoding**: Creates binary columns for each category

**Decision Tree recommendation:**
- **Label Encoding** often works well for Decision Trees!
  Trees can split on any value, so the ordering doesn't matter as much
  as it does for linear models.
- Use **One-Hot Encoding** if a column has no natural order AND
  you want the tree to consider each category independently.

**Example:**
Label Encoding: Color → Red=0, Blue=1, Green=2
Tree can split: "Is Color ≤ 0.5?" → separates Red from Blue/Green""",
        },
        {
            "step": 5,
            "phase": "train_test_split",
            "node_type": "split",
            "title": "Split Data into Train & Test Sets",
            "description": "Divide data for training and evaluation",
            "detailed_instructions": """**Why this matters:**
We train the tree on one portion and test on another to check
if it genuinely learned patterns (not just memorized the data).

**What to do:**
1. Drag the "Train-Test Split" node
2. Connect it after encoding
3. **Select your target column** — what you want to predict
4. Configure split ratio:
   - **80/20**: Standard split (recommended)
   - **70/30**: If you want more test data for evaluation

**Important for Decision Trees:**
Decision Trees are prone to **overfitting** — memorizing training data
instead of learning general patterns. A proper train/test split helps
you detect this.

**Signs of overfitting:**
- Training accuracy: 99% but Test accuracy: 60%
- The tree is very deep with many specific rules
- Solution: Limit max_depth or increase min_samples_split

**For classification targets:** Enable stratified split to keep
class proportions equal in both sets.""",
        },
        {
            "step": 6,
            "phase": "train_model",
            "node_type": "decision_tree",
            "title": "Train Decision Tree Model",
            "description": "Build and train your decision tree",
            "detailed_instructions": """**This is the exciting part!** 🎉

**What happens:**
The algorithm builds the tree by:
1. Finding the best feature and value to split on (using Gini impurity or entropy)
2. Splitting the data into two groups
3. Repeating for each group until stopping criteria are met
4. Creating leaf nodes with final predictions

**What to do:**
1. Drag the "Decision Tree" node from ML Algorithms section
2. Connect it to the train-test split node

**Key hyperparameters — these control tree complexity:**

🌳 **max_depth** (Most important!)
- How many levels deep the tree can grow
- Low (2-3): Simple tree, might underfit
- Medium (5-10): Good balance (recommended start)
- High (20+): Complex tree, risk of overfitting
- **Tip:** Start with 5, increase if accuracy is too low

🍃 **min_samples_split**
- Minimum samples needed to split a node
- Higher values = simpler tree (prevents splitting on tiny groups)
- Default: 2 (try 5-10 to reduce overfitting)

🌱 **min_samples_leaf**
- Minimum samples in each leaf node
- Prevents creating leaves with very few examples
- Default: 1 (try 5-10 for smoother predictions)

**You'll see:**
- Tree visualization (the actual flowchart!)
- Feature importance (which columns matter most)
- Training accuracy or R² score""",
            "success_message": """🎉 **Model Training Complete!**

Your Decision Tree has learned to make predictions from your data!

**Next:** Let's evaluate how well it performs on unseen data.""",
        },
        {
            "step": 7,
            "phase": "evaluate",
            "node_type": "metrics",
            "title": "Evaluate Model Performance",
            "description": "Measure how well the tree predicts on test data",
            "detailed_instructions": """**Understanding Decision Tree Evaluation:**

**For Regression (predicting numbers):**
- **R² Score**: How much variance the tree explains (0-1, higher is better)
  - R² > 0.8: Good model
  - R² 0.5-0.8: Moderate, might need tuning
  - R² < 0.5: Consider deeper tree or more features
- **RMSE**: Average prediction error (lower is better)

**For Classification (predicting categories):**
- Use a **Confusion Matrix** node instead
- Shows True/False Positives and Negatives
- Gives accuracy, precision, recall, and F1 score

**What to do:**
1. Drag the appropriate metric node:
   - "R² Score" and "RMSE" for regression
   - "Confusion Matrix" for classification
2. Connect it to your trained model

**Decision Tree specific insights:**
- Check **feature importance** — which features does the tree use most?
- If test accuracy << train accuracy → tree is overfitting
- **Fix overfitting:** Reduce max_depth or increase min_samples_split

**Comparing with other models:**
- If Decision Tree overfits → try Random Forest (ensemble of trees)
- If accuracy is low → try increasing max_depth or adding features""",
            "success_message": """📊 **Evaluation Complete!**

{metrics_results}

**Analysis:** {performance_analysis}

**Tips for improvement:**
- If overfitting: reduce max_depth, increase min_samples_split/leaf
- If underfitting: increase max_depth, add more features
- Want better accuracy? Try Random Forest (many trees voting together)""",
        },
    ],
    # Completion messages
    "completion": {
        "success": """🎉 **Congratulations!** You've successfully built a Decision Tree model!

**What you accomplished:**
✅ Uploaded and analyzed dataset
✅ Preprocessed data (missing values, encoding)
✅ Split data into train/test sets
✅ Trained a Decision Tree model
✅ Evaluated performance

**What you can do next:**
1. **Adjust max_depth**: Try different depths to find the sweet spot
2. **Check feature importance**: See which features the tree uses most
3. **Visualize the tree**: Understand the rules it learned
4. **Try Random Forest**: Build 100+ trees for better accuracy
5. **Compare models**: Decision Tree vs Linear/Logistic Regression

**Key takeaway:**
Decision Trees are powerful because they're interpretable — you can trace
exactly WHY the model made a prediction. This is invaluable in fields like
healthcare and finance where explainability matters.""",
        "poor_performance": """**Model Training Complete** - Let's improve! 📊

Your model's performance suggests room for improvement. Here's what to try:

**Common Decision Tree fixes:**
1. **Overfitting** (high train accuracy, low test accuracy):
   - Reduce max_depth (try 3-5)
   - Increase min_samples_split (try 10-20)
   - Increase min_samples_leaf (try 5-10)

2. **Underfitting** (low accuracy everywhere):
   - Increase max_depth (try 10-15)
   - Add more features to the dataset
   - Check if the right target column is selected

3. **Try a different algorithm:**
   - Random Forest (ensemble of trees — more robust)
   - MLP Classifier/Regressor (neural network for complex patterns)

**Data quality checks:**
- Do features actually relate to the target?
- Is the dataset large enough? (50+ rows recommended)
- Are there enough examples of each class? (for classification)""",
    },
    "estimated_time": "5-10 minutes",
    "prerequisites": [
        "Dataset in CSV format",
        "Clear target column (categorical or numeric)",
        "Minimum 50 rows of data recommended",
    ],
}
