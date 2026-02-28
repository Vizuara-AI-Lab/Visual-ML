"""
Random Forest Interactive Guide

Comprehensive step-by-step instructions for building a random forest
model with context-aware explanations and dataset guidance.
"""

RANDOM_FOREST_GUIDE = {
    "model_type": "random_forest",
    "title": "Random Forest - Ensemble of Decision Trees",
    # Introduction phase
    "introduction": {
        "title": "What is Random Forest?",
        "explanation": """**Random Forest** builds many decision trees and combines their predictions.
Instead of relying on one tree, it uses the "wisdom of the crowd" — many trees
voting together give better results than any single tree alone.

**How it works:**
1. Create many decision trees (typically 100+)
2. Each tree sees a **random subset** of the data (bagging)
3. Each tree also considers a **random subset** of features at each split
4. For predictions, all trees vote and the majority wins (classification)
   or the average is taken (regression)

**Key components:**
- **Bagging (Bootstrap Aggregating)**: Each tree trains on a random sample of data
- **Feature Randomness**: Each split considers only a subset of features
- **Ensemble Voting**: Final prediction combines all trees' outputs
- **Out-of-Bag (OOB) Score**: Free validation using data each tree didn't see

**Why is it better than one Decision Tree?**
- A single tree might memorize noise → **overfitting**
- Random Forest averages out the noise → **more robust**
- Think of it as: 1 doctor might be wrong, but 100 doctors voting are rarely wrong

**Real-World Examples:**
- 💳 **Fraud Detection**: Is this credit card transaction fraudulent?
- 🏥 **Disease Prediction**: Risk assessment from patient data
- 📊 **Stock Market**: Predicting price movement direction
- 🛒 **Customer Churn**: Will this customer leave?
- 🏠 **House Pricing**: Predicting property values
- 🔬 **Gene Expression**: Identifying important genes from thousands

**When to use Random Forest:**
✅ You want high accuracy with minimal tuning
✅ Your data is tabular (rows and columns)
✅ You want reliable feature importance rankings
✅ You want a model that rarely overfits

**When NOT to use it:**
❌ You need real-time predictions (100+ trees = slower than 1)
❌ You need to explain exactly WHY a prediction was made (use Decision Tree)
❌ You have very high-dimensional sparse data (try linear models)
❌ Your dataset is tiny (<50 rows — not enough for random sampling)""",
        "visual_example": """
Example: Loan Approval with Random Forest

Tree 1 (saw 70% of data):     Tree 2 (saw 70% of data):     Tree 3 (saw 70% of data):
  Income > $45k?                Credit > 680?                  Age > 25?
   /        \\                    /        \\                    /       \\
 Yes        No                 Yes        No                 Yes      No
  |          |                  |          |                  |        |
APPROVE    DENY              APPROVE    DENY              APPROVE   DENY

Final Vote: APPROVE (2) vs DENY (1) → Prediction: APPROVED ✅

Each tree is slightly different because they see different data and features!
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
- Has enough rows — Random Forest benefits from more data
- For **classification**: target should be categorical
- For **regression**: target should be numeric

**Random Forest advantage:** It handles messy real-world data well!
- Mixed feature types (numeric + categorical) ✅
- Some missing values (though we'll clean them) ✅
- Many features (it selects the important ones) ✅

**👉 Drag the 'Upload Dataset' node from the left panel onto the canvas.**""",
                    },
                    {
                        "label": "No, I want to use a sample dataset",
                        "action": "provide_sample",
                        "next_message": """Perfect! Here are great datasets for learning Random Forest:

**For Regression (predicting numbers):**
1. 🏠 **Housing** - Predict house prices (Random Forest excels here!)
2. 🚗 **Auto MPG** - Predict fuel efficiency
3. 💰 **Tips** - Predict restaurant tip amounts

**For Classification (predicting categories):**
4. 🚢 **Titanic** - Predict survival (great for seeing feature importance)
5. 🏥 **Breast Cancer** - Classify tumors (high accuracy expected!)
6. ❤️ **Heart Disease** - Predict heart disease presence

**Tip:** Random Forest typically outperforms a single Decision Tree on
all of these datasets. Try both and compare!

Which would you like to try? (Or upload your own dataset)""",
                    },
                    {
                        "label": "I want to learn more about datasets first",
                        "action": "explain_dataset",
                        "next_message": """**Understanding Datasets for Random Forest:**

A dataset is a table where each row is one example and columns are features.

**Example: House Pricing**
```
| Size  | Bedrooms | Location  | Age | Price    |
|-------|----------|-----------|-----|----------|
| 1500  | 3        | Suburban  | 10  | $250,000 |
| 2200  | 4        | Urban     | 5   | $450,000 |
| 800   | 1        | Rural     | 30  | $120,000 |
```

**How Random Forest uses this:**
- Tree 1 might focus on Size and Location
- Tree 2 might focus on Bedrooms and Age
- Tree 3 might focus on Size and Age
- Together, they capture ALL the important patterns!

**Feature importance output:** Random Forest tells you which features
matter most. For example: Size (45%), Location (30%), Age (15%), Bedrooms (10%)

**Key advantage over single Decision Tree:**
- Decision Tree might overfit to noise in the training data
- Random Forest averages many trees → cancels out the noise → better predictions

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

**Tips for Random Forest:**
- More data = better! Random Forest thrives with larger datasets
- Don't worry about feature scaling — trees don't need it
- Many features? Great! Random Forest handles high-dimensional data well""",
            "validation_checks": [
                "Dataset has been uploaded successfully",
                "Dataset contains a clear target column",
                "Dataset has at least 2 feature columns",
            ],
            "common_errors": {
                "too_few_rows": "Small datasets limit Random Forest's potential. Aim for 100+ rows.",
                "single_column": "Need at least 2 feature columns for the forest to use random feature selection.",
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
- Classification or regression target?
- Distribution of values
- Class balance (for classification)

🌲 **Random Forest Specific Checks:**
- Number of features (affects max_features parameter)
- Dataset size (affects n_estimators recommendation)
- Feature types for encoding recommendations""",
        },
        {
            "step": 3,
            "phase": "handle_missing_values",
            "node_type": "missing_value_handler",
            "title": "Handle Missing Values",
            "description": "Clean missing/null values in your dataset",
            "skip_condition": "no_missing_values",
            "detailed_instructions": """**Why this matters:**
While individual trees can sometimes handle missing data, cleaning it
ensures consistent and reliable results across all trees in the forest.

**What to do:**
1. Drag the "Handle Missing Values" node from the left panel
2. Connect it to your uploaded dataset node
3. Choose a strategy:
   - **Median**: For numeric columns (robust to outliers)
   - **Mode**: For categorical columns (most common value)
   - **Drop Rows**: If very few rows have missing values (<5%)

**Random Forest tip:**
Median imputation works well because trees split on individual values.
The tree can learn to split around the imputed value if needed.

**⚠️ Don't** drop too many rows — Random Forest needs sufficient data
for its random sampling (bagging) to be effective.""",
        },
        {
            "step": 4,
            "phase": "encoding",
            "node_type": "encoding",
            "title": "Encode Categorical Features",
            "description": "Convert text/category columns to numbers",
            "skip_condition": "no_categorical_columns",
            "detailed_instructions": """**Why this matters:**
Random Forest (like all scikit-learn models) needs numeric input.
We convert categorical text values to numbers.

**What to do:**
1. Drag the "Encoding" node from the preprocessing section
2. Connect it after the missing value handler
3. Choose encoding method per column:
   - **Label Encoding**: Assigns numbers to categories (Red=0, Blue=1)
   - **One-Hot Encoding**: Creates binary columns per category

**Random Forest recommendation:**
- **Label Encoding** usually works great for Random Forest!
  Since trees split on thresholds, they can handle label-encoded
  values effectively regardless of the assigned numbers.
- **One-Hot Encoding** is better if categories have no inherent order
  AND you have few categories (<10 per column).
- **Avoid** One-Hot for high-cardinality columns (50+ categories) —
  it creates too many features.

**Why trees handle Label Encoding well:**
Tree: "Is Color ≤ 1?" → This separates {Red=0, Blue=1} from {Green=2}
Another tree: "Is Color ≤ 0?" → This separates {Red=0} from {Blue=1, Green=2}
The forest explores many possible splits!""",
        },
        {
            "step": 5,
            "phase": "train_test_split",
            "node_type": "split",
            "title": "Split Data into Train & Test Sets",
            "description": "Divide data for training and evaluation",
            "detailed_instructions": """**Why this matters:**
Even though Random Forest has its own internal validation (OOB score),
a proper train/test split gives you an unbiased performance estimate.

**What to do:**
1. Drag the "Train-Test Split" node
2. Connect it after encoding
3. **Select your target column** — what you want to predict
4. Configure split ratio:
   - **80/20**: Standard split (recommended)
   - **70/30**: Good if you have a large dataset

**Random Forest and overfitting:**
Random Forest is much less prone to overfitting than a single Decision Tree.
However, a train/test split is still essential to verify this!

**Good signs:**
- Train and test scores are close → model generalizes well
- This is the typical outcome with Random Forest!

**For classification:** Enable stratified split to keep class
proportions balanced in both train and test sets.""",
        },
        {
            "step": 6,
            "phase": "train_model",
            "node_type": "random_forest",
            "title": "Train Random Forest Model",
            "description": "Build and train your ensemble of decision trees",
            "detailed_instructions": """**This is the exciting part!** 🎉

**What happens:**
The algorithm builds many trees, each slightly different:
1. For each tree: randomly sample ~70% of the data (with replacement)
2. At each split: consider only a random subset of features
3. Grow each tree to its maximum depth (or limited by hyperparameters)
4. Combine all trees: majority vote (classification) or average (regression)

**What to do:**
1. Drag the "Random Forest" node from ML Algorithms section
2. Connect it to the train-test split node

**Key hyperparameters:**

🌲 **n_estimators** (Number of trees — MOST important!)
- How many trees to build
- More trees = better accuracy but slower
- **100**: Good default starting point
- **200-500**: Try if accuracy is too low
- Diminishing returns after ~300 trees usually

🌳 **max_depth** (Tree depth limit)
- Limits how deep each individual tree can grow
- **None** (default): Trees grow fully — usually fine for Random Forest!
- **10-20**: If you want faster training
- Unlike single Decision Trees, full-depth trees rarely overfit in a forest

🍃 **min_samples_split**
- Minimum samples to split a node
- Default: 2 (usually fine for Random Forest)
- Increase to 5-10 for very noisy data

🌱 **min_samples_leaf**
- Minimum samples in each leaf
- Default: 1 (fine for most cases)
- Increase to 3-5 for smoother predictions

**Pro tip:** Random Forest usually works great with defaults!
Start with n_estimators=100 and only tune if accuracy is insufficient.

**You'll see:**
- Feature importance rankings (very reliable in Random Forest)
- Training accuracy/R² score
- OOB score (if available)""",
            "success_message": """🎉 **Model Training Complete!**

Your Random Forest has built an ensemble of decision trees!

**Next:** Let's evaluate how well the forest predicts on unseen data.""",
        },
        {
            "step": 7,
            "phase": "evaluate",
            "node_type": "metrics",
            "title": "Evaluate Model Performance",
            "description": "Measure how well the forest predicts on test data",
            "detailed_instructions": """**Understanding Random Forest Evaluation:**

**For Regression (predicting numbers):**
- **R² Score**: Proportion of variance explained (0-1, higher is better)
  - R² > 0.85: Excellent (Random Forest often achieves this!)
  - R² 0.7-0.85: Good
  - R² < 0.7: Try more trees or check your features
- **RMSE**: Average prediction error in original units (lower is better)

**For Classification (predicting categories):**
- Use **Confusion Matrix** node for detailed class-level metrics
- **Accuracy**: Overall correct predictions
- **Precision/Recall/F1**: Per-class metrics

**What to do:**
1. Drag the appropriate metric node:
   - "R² Score" and "RMSE" for regression
   - "Confusion Matrix" for classification
2. Connect it to your trained model

**Random Forest insights to look for:**
📊 **Feature Importance**: The most reliable insight from Random Forest!
- Shows which features the model relies on most
- Use this to understand your data
- Consider dropping low-importance features

📈 **Train vs Test comparison:**
- Close scores → great generalization (typical for Random Forest)
- Large gap → possible overfitting (rare — try reducing max_depth)

**Comparing with Decision Tree:**
Random Forest usually scores 5-15% higher than a single Decision Tree.
If not, your data might be simple enough for a single tree.""",
            "success_message": """📊 **Evaluation Complete!**

{metrics_results}

**Analysis:** {performance_analysis}

**Tips for improvement:**
- Increase n_estimators (more trees = better, up to a point)
- Check feature importance — remove irrelevant features
- Try different max_depth values if accuracy plateaus
- Compare with a single Decision Tree to see the ensemble benefit""",
        },
    ],
    # Completion messages
    "completion": {
        "success": """🎉 **Congratulations!** You've successfully built a Random Forest model!

**What you accomplished:**
✅ Uploaded and analyzed dataset
✅ Preprocessed data (missing values, encoding)
✅ Split data into train/test sets
✅ Trained a Random Forest ensemble
✅ Evaluated performance

**What you can do next:**
1. **Check feature importance**: See which features drive predictions
2. **Tune n_estimators**: Try 200 or 500 trees for better accuracy
3. **Compare models**: Random Forest vs Decision Tree — see the improvement!
4. **Try MLP**: Neural network might capture patterns trees miss
5. **Experiment with features**: Remove low-importance features and retrain

**Key takeaway:**
Random Forest is one of the most reliable ML algorithms for tabular data.
It combines the interpretability of decision trees with the robustness of
ensemble methods. Feature importance from Random Forest is widely used
in real-world data analysis and feature engineering.""",
        "poor_performance": """**Model Training Complete** - Let's improve! 📊

Your model could perform better. Here's what to try:

**Random Forest specific improvements:**
1. **Increase n_estimators**: Try 200-500 trees (more trees rarely hurts)
2. **Adjust max_depth**: Try None (unlimited) to let trees grow fully
3. **Check feature importance**: Are relevant features in your dataset?

**Data improvements:**
4. **More data**: Random Forest benefits greatly from larger datasets
5. **Feature engineering**: Create new features from existing ones
6. **Check target variable**: Is it the right column? Right type?

**Alternative algorithms:**
- **MLP Classifier/Regressor**: Neural network for complex non-linear patterns
- **Logistic Regression**: If the relationship is actually linear (simpler is better!)

**Common issues:**
- Very small dataset (<100 rows): Not enough for random sampling
- Too few features: Forest can't differentiate with random feature subsets
- Noisy target: Check if your target variable is reliable""",
    },
    "estimated_time": "5-10 minutes",
    "prerequisites": [
        "Dataset in CSV format",
        "Clear target column (categorical or numeric)",
        "Minimum 100 rows of data recommended for Random Forest",
    ],
}
