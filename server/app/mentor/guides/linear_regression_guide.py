"""
Linear Regression Interactive Guide

Comprehensive step-by-step instructions for building a linear regression model
with context-aware explanations and dataset guidance.
"""

LINEAR_REGRESSION_GUIDE = {
    "model_type": "linear_regression",
    "title": "Linear Regression - Predict Continuous Values",
    # Introduction phase
    "introduction": {
        "title": "What is Linear Regression?",
        "explanation": """**Linear Regression** is a fundamental machine learning algorithm used to predict 
continuous numeric values (like prices, temperatures, sales, etc.) based on input features.

**How it works:**
Linear regression finds the best-fitting straight line (or hyperplane in multiple dimensions) through your data. 
It assumes a linear relationship between:
- **Features (X)**: Input variables (e.g., house size, number of bedrooms)
- **Target (y)**: Output variable you want to predict (e.g., house price)

**Real-World Examples:**
- 🏠 **House Price Prediction**: Predict house prices based on size, location, bedrooms
- 📈 **Sales Forecasting**: Predict future sales based on advertising spend, seasonality
- 🌡️ **Temperature Prediction**: Predict temperature based on humidity, wind speed
- 💰 **Salary Estimation**: Predict salary based on years of experience, education level
- 🚗 **Fuel Efficiency**: Predict car mileage based on engine size, weight

**When to use Linear Regression:**
✅ Your target variable is continuous (numbers, not categories)
✅ You expect a roughly linear relationship between features and target
✅ You want an interpretable model that explains feature importance

**When NOT to use it:**
❌ For classification tasks (use Logistic Regression instead)
❌ When relationships are highly non-linear (try Decision Trees/Random Forest)
❌ With too many outliers (consider robust regression)
""",
        "visual_example": """
Example: Predicting House Prices
Feature (Size) →  Target (Price)
1000 sq ft    →  $200,000
1500 sq ft    →  $280,000
2000 sq ft    →  $360,000
2500 sq ft    →  $440,000

The algorithm finds: Price = $80 + (Size × $160)
So a 1800 sq ft house → $80 + (1800 × $160) = $368,000
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
- Has at least one numeric column for the target variable
- Has numeric or categorical features
- Doesn't have too many missing values (we can handle some)

**👉 Look at the left panel under 'Data Source' and drag the 'Upload Dataset' node to the canvas.**""",
                    },
                    {
                        "label": "No, I want to use a sample dataset",
                        "action": "provide_sample",
                        "next_message": """Perfect! I'll help you with a sample dataset. Here are some options:

**Sample Datasets for Linear Regression:**
1. 🏠 **Housing Prices** - Predict house prices from size, bedrooms, location
2. 📊 **Student Performance** - Predict test scores from study hours
3. 🚗 **Car Mileage** - Predict fuel efficiency from car specifications
4. 💼 **Salary Prediction** - Predict salary from experience and education

Which would you like to try? (Or upload your own dataset using the 'Upload Dataset' node)""",
                    },
                    {
                        "label": "I want to learn more about datasets first",
                        "action": "explain_dataset",
                        "next_message": """**Understanding Datasets for Linear Regression:**

A dataset is a table with:
- **Rows**: Individual examples (e.g., different houses, students, cars)
- **Columns**: Features and target variable

**Example Dataset Structure:**
```
| Size(sqft) | Bedrooms | Age(years) | Price($) |
|------------|----------|------------|----------|
| 1500       | 3        | 10         | 280000   | ← One row = one house
| 2000       | 4        | 5          | 360000   |
| 1200       | 2        | 15         | 220000   |
```

**What you need:**
- **Features (X)**: The columns you'll use to make predictions (Size, Bedrooms, Age)
- **Target (y)**: The column you want to predict (Price)
- **Enough data**: At least 50-100 rows for good results

**Ready to continue?** Drag the 'Upload Dataset' node from the left panel.""",
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
            "description": "Load your CSV file containing features and target variable",
            "detailed_instructions": """**What to do:**
1. Look at the **left sidebar** under "Data Source" section
2. **Drag** the "Upload Dataset" node onto the canvas
3. **Click** on the node to configure it
4. **Browse** and select your CSV file
5. Click "Upload" and wait for processing

**What happens:**
- Your data is uploaded and validated
- I'll analyze the columns and data types
- You'll see a preview of your dataset
- I'll give you insights and recommendations

**Tips:**
- Supported format: CSV files only
- File size limit: 100MB
- Make sure the first row contains column names""",
            "validation_checks": [
                "Dataset has been uploaded successfully",
                "Dataset contains at least one numeric column",
                "Dataset has more than 10 rows",
            ],
            "common_errors": {
                "file_too_large": "File is too large. Try a smaller dataset or sample your data.",
                "invalid_format": "File must be CSV format. Convert Excel files to CSV first.",
                "no_numeric_columns": "Dataset needs numeric columns for linear regression.",
            },
        },
        {
            "step": 2,
            "phase": "data_analysis",
            "node_type": "data_analysis",
            "title": "Analyze Dataset Quality",
            "description": "I'll examine your data and suggest preprocessing steps",
            "automated": True,
            "detailed_instructions": """**I'm analyzing your dataset for:**

📊 **Data Quality Checks:**
- Missing values count and percentage
- Data types of each column
- Numeric vs categorical features
- Outliers and unusual values
- Column statistics (mean, min, max, std)

🎯 **Target Variable Identification:**
- Looking for likely target columns
- Checking if it's numeric (required for regression)
- Analyzing target distribution

📈 **Feature Analysis:**
- Identifying categorical columns that need encoding
- Finding features that need scaling
- Detecting high-cardinality issues

I'll tell you what preprocessing steps are needed next!""",
            "success_message": """✅ **Dataset Analysis Complete!**

I've examined your data. Here's what I found:
{analysis_results}

**Recommended next steps:** {recommendations}""",
        },
        {
            "step": 3,
            "phase": "handle_missing_values",
            "node_type": "missing_value_handler",
            "title": "Handle Missing Values",
            "description": "Clean missing/null values in your dataset",
            "skip_condition": "no_missing_values",
            "detailed_instructions": """**Why this matters:**
Missing values (NaN, null, empty) can cause errors in machine learning models.

**What to do:**
1. Drag the "Handle Missing Values" node from the left panel
2. Connect it to your uploaded dataset node
3. Choose a strategy:
   - **Mean/Median**: For numeric columns (recommended for most cases)
   - **Mode**: For categorical columns
   - **Drop Rows**: If few rows have missing values (<5%)
   - **Drop Columns**: If column has too many missing values (>50%)

**My recommendation:** {missing_value_recommendation}

**Tip:** For numeric data, median is more robust to outliers than mean.""",
        },
        {
            "step": 4,
            "phase": "encoding",
            "node_type": "encoding",
            "title": "Encode Categorical Features",
            "description": "Convert text/category columns to numbers",
            "skip_condition": "no_categorical_columns",
            "detailed_instructions": """**Why this matters:**
Machine learning models only understand numbers, not text or categories.

**What to do:**
1. Drag the "Encoding" node from the preprocessing section
2. Connect it after the missing value handler
3. Select categorical columns to encode
4. Choose encoding method:
   - **Label Encoding**: For ordinal data (e.g., Small=0, Medium=1, Large=2)
   - **One-Hot Encoding**: For nominal data (e.g., Color: Red, Blue, Green)

**Your categorical columns:** {categorical_columns}

**My recommendation:** {encoding_recommendation}

**Tip:** One-hot encoding is safer when categories have no natural order.""",
        },
        {
            "step": 5,
            "phase": "scaling",
            "node_type": "scaling",
            "title": "Scale Features (Optional but Recommended)",
            "description": "Normalize feature values to similar ranges",
            "optional": True,
            "detailed_instructions": """**Why this matters:**
Features with larger values can dominate the model. Scaling puts all features on equal footing.

**Example:**
- House Size: 500-5000 (large range)
- Bedrooms: 1-5 (small range)
→ Without scaling, size will have more influence

**What to do:**
1. Drag the "Scaling" node from preprocessing
2. Connect it after encoding
3. Choose a method:
   - **StandardScaler**: Centers around 0, most common (recommended)
   - **MinMaxScaler**: Scales to 0-1 range
   - **RobustScaler**: Best if you have outliers

**My recommendation:** {scaling_recommendation}

**Note:** Scaling is optional but usually improves results.""",
        },
        {
            "step": 6,
            "phase": "train_test_split",
            "node_type": "split",
            "title": "Split Data into Train & Test Sets",
            "description": "Divide data for training and evaluation",
            "detailed_instructions": """**Why this matters:**
We need separate data to:
- **Training set**: Teach the model (typically 70-80% of data)
- **Test set**: Evaluate how well it learned (20-30% of data)

This prevents "cheating" - the model must predict on data it hasn't seen before!

**What to do:**
1. Drag the "Train-Test Split" node
2. Connect it after scaling (or encoding if you skipped scaling)
3. Configure split ratio:
   - **80/20**: Standard split (recommended)
   - **70/30**: If you have less data
   - **90/10**: If you have lots of data

**Tip:** Keep test set at least 20% to get reliable evaluation.""",
        },
        {
            "step": 7,
            "phase": "train_model",
            "node_type": "linear_regression",
            "title": "Train Linear Regression Model",
            "description": "Build and train your prediction model",
            "detailed_instructions": """**This is the exciting part!** 🎉

**What happens:**
The algorithm finds the best-fitting line through your data by:
1. Starting with random coefficients
2. Measuring prediction errors
3. Adjusting coefficients to minimize errors
4. Repeating until optimal

**What to do:**
1. Drag the "Linear Regression" node from ML Algorithms section
2. Connect it to the train-test split node
3. Select your **target column** (what you want to predict)
4. Select **feature columns** (what you'll use to predict)
5. Click "Train Model"

**Configuration tips:**
- **fit_intercept**: Keep this ON (includes bias term)
- **normalize**: Turn OFF if you already scaled data
- **n_jobs**: Use -1 for faster training (uses all CPU cores)

**You'll see:**
- Model coefficients (how much each feature influences prediction)
- Training progress
- Initial performance metrics""",
            "success_message": """🎉 **Model Training Complete!**

Your linear regression model has learned the patterns in your data!

**Next:** Let's evaluate how well it performs on unseen test data.""",
        },
        {
            "step": 8,
            "phase": "evaluate",
            "node_type": "metrics",
            "title": "Evaluate Model Performance",
            "description": "Measure prediction accuracy with R², RMSE, MAE",
            "detailed_instructions": """**Understanding Performance Metrics:**

We use 3 key metrics for regression:

1. **R² Score (R-squared)** - Most important! 📊
   - Range: 0 to 1 (or negative if very bad)
   - **0.9-1.0**: Excellent! Model explains 90%+ of variance
   - **0.7-0.9**: Good performance
   - **0.5-0.7**: Moderate, maybe try other algorithms
   - **<0.5**: Poor, need better features or different approach

2. **RMSE (Root Mean Squared Error)**
   - Average prediction error in same units as target
   - **Lower is better**
   - Example: RMSE=$20,000 means predictions are off by $20k on average

3. **MAE (Mean Absolute Error)**
   - Similar to RMSE but less sensitive to outliers
   - **Lower is better**

**What to do:**
1. Drag the "Metrics" node from Results section
2. Connect it to your trained model
3. View the results

**Interpreting results:** I'll help you understand if your model is good enough!""",
            "success_message": """📈 **Evaluation Results:**

{metrics_results}

**Analysis:** {performance_analysis}

**Next steps:** {next_recommendations}""",
        },
    ],
    # Completion messages
    "completion": {
        "success": """🎉 **Congratulations!** You've successfully built a Linear Regression model!

**What you accomplished:**
✅ Uploaded and analyzed dataset
✅ Preprocessed data (handling missing values, encoding, scaling)
✅ Split data into train/test sets
✅ Trained a linear regression model
✅ Evaluated performance with metrics

**What you can do next:**
1. **Try different features**: Remove or add features to improve R²
2. **Experiment with preprocessing**: Try different scaling methods
3. **Deploy your model**: Use it to make predictions on new data
4. **Try other algorithms**: Compare with Decision Tree or Random Forest
5. **Visualize predictions**: See how predictions match actual values

**Want to learn more?**
- How linear regression coefficients work
- Feature importance analysis
- Residual analysis
- Prediction confidence intervals

Just ask me anything!""",
        "poor_performance": """**Model Training Complete** - But we can improve! 📊

Your R² score suggests the model could be better. Here's what we can try:

**Common improvements:**
1. **Feature Engineering**: Create new features from existing ones
2. **Remove outliers**: Clean extreme values that skew the model
3. **Try polynomial features**: Capture non-linear relationships
4. **Different algorithm**: Linear Regression might not be the best fit
   - Try: Decision Tree, Random Forest, or Gradient Boosting

**Data quality checks:**
- Is your target variable truly continuous?
- Do features have a linear relationship with target?
- Is there enough data (ideally 100+ rows)?

**I can help you:**
- Analyze which features are most important
- Detect and handle outliers
- Try a different ML algorithm
- Improve data preprocessing

What would you like to do next?""",
    },
    # Estimated time
    "estimated_time": "5-10 minutes",
    # Prerequisites
    "prerequisites": [
        "Dataset in CSV format",
        "At least one numeric target column",
        "Minimum 50 rows of data recommended",
    ],
}
