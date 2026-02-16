"""
Node Explanations with Simple, Everyday Life Examples

Each node's purpose explained in simple terms with relatable, day-to-day examples
to help beginners understand machine learning concepts easily.
"""

from typing import Dict

# Node explanations with simple everyday examples
NODE_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "upload_file": {
        "simple": "Upload your data file to start working",
        "example": "Think of this like uploading photos to Google Photos - you're giving the system your information to work with. Just like you need photos before editing them, you need data before building a model.",
        "full": "This is where you give the system your data - like a CSV file with information. It's the starting point, just like you need ingredients before cooking a meal.",
    },
    "missing_value_handler": {
        "simple": "Fix empty or missing cells in your data",
        "example": "Imagine a form where some people didn't fill in their age. You can either: 1) Remove those incomplete forms, 2) Fill in the average age, or 3) Put 'Unknown'. This node does the same for your data - it handles those blank spots automatically.",
        "full": "Your data might have empty cells (like missing phone numbers in a contact list). This node helps you decide what to do with them - remove those rows, fill with average values, or use smart guessing based on other data.",
    },
    "encoding": {
        "simple": "Convert text labels into numbers",
        "example": "Computers don't understand 'Red', 'Blue', 'Green' - they only understand numbers. This is like translating colors to numbers: Red=1, Blue=2, Green=3. So if you have a column with fruit names (Apple, Banana, Orange), this node converts them to numbers (1, 2, 3) that the computer can process.",
        "full": "Machine learning models need numbers to work with. This node converts text categories (like 'Male'/'Female' or 'Yes'/'No') into numerical codes that the computer can understand and calculate with.",
    },
    "scaling": {
        "simple": "Make all numbers use the same range",
        "example": "Imagine comparing salaries ($30,000) with ages (30 years). The salary number is huge compared to age! Scaling is like converting everything to a 0-10 scale, so $30,000 might become 5, and age 30 might become 6. Now they're fair to compare - like giving everyone the same ruler to measure with.",
        "full": "When you have columns with very different number ranges (like salary in thousands vs age in 2-digits), this node adjusts them to similar scales. It's like converting Celsius and Fahrenheit to the same temperature scale for fair comparison.",
    },
    "feature_selection": {
        "simple": "Pick only the most important columns",
        "example": "When predicting house prices, knowing the number of bedrooms matters, but knowing the owner's favorite color doesn't. This is like packing for a trip - you don't take everything, just what's useful. This node identifies and keeps only the helpful information, leaving out the noise.",
        "full": "Not all data columns are equally useful for predictions. This node analyzes which features (columns) actually help make better predictions and removes the rest - like choosing only the relevant ingredients for a recipe.",
    },
    "split": {
        "simple": "Divide your data into practice set and test set",
        "example": "Think of studying for an exam: you practice with some questions (training set) and test yourself with different questions (test set). You don't memorize the test questions! Similarly, we use 80% of data to teach the model and keep 20% hidden to test how well it really learned.",
        "full": "Splits your dataset into two parts: Training data (usually 80%) to teach the model, and Testing data (20%) to check if it learned properly. Like a teacher using some questions for teaching and others for the final exam.",
    },
    "linear_regression": {
        "simple": "Predict numbers (like prices, temperatures, scores)",
        "example": "Remember drawing a 'line of best fit' in math class through scattered points? That's exactly what this does! If you plot 'study hours vs exam scores', this finds the best line and uses it to predict: 'If I study 5 hours, what score will I get?' Perfect for predicting house prices, sales, temperatures, etc.",
        "full": "This model finds the mathematical relationship between features and a continuous number output. It's like finding the formula: Price = (Bedrooms × $50k) + (Area × $100) + Base Price.",
    },
    "logistic_regression": {
        "simple": "Predict yes/no or category choices",
        "example": "Instead of predicting a number, this predicts categories. Like an email filter deciding 'Spam or Not Spam?' or a doctor predicting 'Disease or Healthy?' based on symptoms. If 3 symptoms are present, it calculates: '85% chance of disease' and says 'Yes, likely has disease'.",
        "full": "Predicts categories or yes/no outcomes by calculating probabilities. Unlike Linear Regression (which predicts numbers), this predicts which category something belongs to - perfect for classification tasks.",
    },
    "decision_tree": {
        "simple": "Makes decisions like a flowchart",
        "example": "Imagine deciding what to wear: 'Is it cold? → Yes → Is it raining? → Yes → Wear warm raincoat'. The model builds a tree of questions like this. For example, to predict if someone will buy a product: 'Age > 30? → Income > $50k? → Previous buyer? → YES, will buy!'",
        "full": "Creates a tree of questions (like a game of 20 questions) to make predictions. Each branch asks a yes/no question about your data until it reaches a decision at the bottom.",
    },
    "random_forest": {
        "simple": "Ask multiple decision trees and take a vote",
        "example": "Instead of trusting one expert, you ask 100 experts and take majority vote! Each tree makes a prediction, then they vote. Like asking 100 doctors if you're healthy - if 85 say 'yes', you trust that answer. More reliable than a single decision tree.",
        "full": "Builds many decision trees (a 'forest') and combines their predictions. Like getting multiple opinions before making an important decision - more accurate and less likely to make mistakes than a single tree.",
    },
    "view_data": {
        "simple": "See your data in a table",
        "example": "Like opening an Excel spreadsheet to see what's inside. You can see all the rows and columns, just like viewing your contacts list or a shopping cart. Helpful to check if your data looks correct.",
        "full": "Displays your dataset in a table format so you can see the actual numbers and text. Essential for checking if data was loaded correctly and understanding what you're working with.",
    },
    "data_preview": {
        "simple": "Quick peek at first few rows",
        "example": "Like when Netflix shows you the first 30 seconds of a show - you get a preview without watching everything. This shows you the first 5-10 rows of your data so you know what it looks like without scrolling through thousands of rows.",
        "full": "Shows a preview of your dataset (first few rows) to quickly understand its structure without loading the entire dataset. Like previewing a document before downloading it.",
    },
    "column_info": {
        "simple": "Details about each column in your data",
        "example": "Like a nutrition label on food - it tells you what's in each column: 'Age column has numbers from 18-65, Name column has text, Salary column has numbers from $30k-$150k'. Helps you understand your data's ingredients.",
        "full": "Provides detailed statistics about each column: data type (number/text), range of values, unique values, and missing data count. Essential for understanding your dataset's structure.",
    },
    "statistics_view": {
        "simple": "Mathematical summary of your numbers",
        "example": "Like checking your phone's battery stats: 'Average battery life: 12 hours, Maximum: 15 hours, Minimum: 8 hours'. This calculates average, min, max, and median for all number columns in your data.",
        "full": "Calculates statistical measures (mean, median, standard deviation, min, max) for numerical columns. Helps you understand the distribution and range of your data.",
    },
    "chart_view": {
        "simple": "Visualize your data as graphs and charts",
        "example": "Instead of seeing a list of temperatures, you see a graph showing how temperature changed over days. Like seeing your weight loss progress on a chart instead of just numbers - much easier to understand patterns!",
        "full": "Creates visual representations (bar charts, line graphs, scatter plots) of your data. Makes patterns and relationships much easier to spot than looking at raw numbers.",
    },
    "predictions": {
        "simple": "Use trained model to predict new data",
        "example": "After teaching the model with examples, now you can ask it to predict new cases. Like after your GPS learns your route to work, it can predict traffic for tomorrow. Give it new house features → it predicts the price!",
        "full": "Uses your trained model to make predictions on new, unseen data. The model applies what it learned during training to make predictions on fresh examples.",
    },
    "metrics": {
        "simple": "Check how good your model performed",
        "example": "Like checking your exam score: 'You got 85% correct'. This tells you accuracy, errors, and how well your model works. If accuracy is 60%, your model is average. If 95%, it's excellent!",
        "full": "Evaluates your model's performance using metrics like accuracy, precision, recall, and error rates. Tells you how well your model is doing at making predictions.",
    },
    "confusion_matrix": {
        "simple": "See where your model made mistakes",
        "example": "Like a teacher marking 'Correct', 'Wrong-thought YES but was NO', 'Wrong-thought NO but was YES'. Shows exactly how many the model got right vs wrong, and what kind of mistakes it made. Great for understanding errors!",
        "full": "A table showing correct predictions and mistakes your model made. Shows True Positives, False Positives, True Negatives, and False Negatives - helps you understand exactly where the model is confused.",
    },
    "feature_importance": {
        "simple": "Which factors matter most for predictions",
        "example": "When predicting house prices, it might show: 'Location matters 40%, Size matters 35%, Age matters 15%, Color matters 2%'. Now you know location is the most important! Like knowing which exam topics carry the most marks.",
        "full": "Shows which features (columns) had the biggest impact on the model's predictions. Helps you understand what factors are driving the results.",
    },
}


def get_node_explanation(node_type: str, format: str = "full") -> str:
    """
    Get explanation for a node type.

    Args:
        node_type: Type of the node
        format: 'simple', 'example', or 'full'

    Returns:
        Explanation string
    """
    node_info = NODE_EXPLANATIONS.get(node_type, {})

    if format == "simple":
        return node_info.get("simple", f"This is a {node_type.replace('_', ' ')} node.")
    elif format == "example":
        return node_info.get("example", "")
    elif format == "full":
        simple = node_info.get("simple", "")
        example = node_info.get("example", "")
        full = node_info.get("full", "")

        if example:
            return f"{simple}\n\n**Real-Life Example:**\n{example}\n\n{full}"
        else:
            return full or simple or f"This is a {node_type.replace('_', ' ')} node."

    return node_info.get(format, "")


def get_simple_explanation_for_message(node_type: str) -> str:
    """
    Get a simple explanation suitable for mentor messages.

    Args:
        node_type: Type of the node

    Returns:
        Combined simple explanation with example
    """
    node_info = NODE_EXPLANATIONS.get(node_type, {})
    simple = node_info.get("simple", "")
    example = node_info.get("example", "")

    if simple and example:
        return f"{simple}. {example}"
    elif simple:
        return simple
    elif example:
        return example
    else:
        return f"This helps with {node_type.replace('_', ' ')}."
