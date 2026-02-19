/**
 * Interactive Data Storytelling — narrative configs for sample datasets.
 * Each story guides the student through understanding the dataset with
 * a character-driven narrative, step-by-step guidance, and a challenge.
 */

export interface StoryStep {
  id: string;
  title: string;
  description: string;
  hint?: string;
  /** Node type the student should add/configure for this step */
  nodeType?: string;
}

export interface DatasetStory {
  datasetId: string;
  title: string;
  emoji: string;
  color: string;
  narrative: string;
  characterContext: string;
  steps: StoryStep[];
  challenge: {
    description: string;
    metric: string;
    threshold: string;
  };
}

export const datasetStories: Record<string, DatasetStory> = {
  iris: {
    datasetId: "iris",
    title: "The Iris Garden Mystery",
    emoji: "🌸",
    color: "#8B5CF6",
    narrative:
      "You're a botanist visiting a garden with 150 iris flowers from 3 species. Your goal: build a classifier that identifies species from petal and sepal measurements alone. Can you train a model that never misidentifies a flower?",
    characterContext: "Dr. Flora, a seasoned botanist, guides you through the garden.",
    steps: [
      {
        id: "iris-1",
        title: "Explore the Garden",
        description:
          "Start by loading the Iris dataset and examining the data. How many features are there? What do they represent?",
        hint: "Add a Sample Dataset node and connect a Table View or Statistics View to inspect the data.",
        nodeType: "sample_dataset",
      },
      {
        id: "iris-2",
        title: "Visualize the Patterns",
        description:
          "Plot the data to see if species form distinct clusters. Which pair of measurements separates species best?",
        hint: "Connect a Chart View node to see scatter plots of the features.",
        nodeType: "chart_view",
      },
      {
        id: "iris-3",
        title: "Prepare for Training",
        description:
          "Split the data into training and testing sets. Use 80% for training so the model has enough examples to learn from.",
        hint: "Add a Split node with 80/20 ratio.",
        nodeType: "split",
      },
      {
        id: "iris-4",
        title: "Train Your Classifier",
        description:
          "Train a Logistic Regression model on the training set. This classic algorithm works well for linearly separable data like Iris.",
        hint: "Connect a Logistic Regression node to the training output of Split.",
        nodeType: "logistic_regression",
      },
      {
        id: "iris-5",
        title: "Evaluate Your Model",
        description:
          "Check how well your model performs on unseen data using a Confusion Matrix. Can you achieve near-perfect accuracy?",
        hint: "Add a Confusion Matrix node connected to the model output.",
        nodeType: "confusion_matrix",
      },
    ],
    challenge: {
      description:
        "Build a pipeline that correctly classifies at least 95% of the test Iris flowers.",
      metric: "Accuracy",
      threshold: "95%",
    },
  },

  titanic: {
    datasetId: "titanic",
    title: "Surviving the Titanic",
    emoji: "🚢",
    color: "#3B82F6",
    narrative:
      "It's 1912 and the Titanic has struck an iceberg. Using passenger data — class, age, fare, family size — predict who survived. Discover how social class and gender affected survival odds in this historical tragedy.",
    characterContext:
      "Captain Data, a maritime historian, shares insights about the passengers.",
    steps: [
      {
        id: "titanic-1",
        title: "Board the Ship",
        description:
          "Load the Titanic dataset and preview the data. Notice which columns have missing values — this is a real-world dataset!",
        hint: "Use Sample Dataset → Table View to explore the raw data.",
        nodeType: "sample_dataset",
      },
      {
        id: "titanic-2",
        title: "Handle Missing Passengers",
        description:
          "Some ages and cabin numbers are missing. Handle these missing values — try filling age with the median value.",
        hint: "Add a Missing Value Handler node and configure it for the Age column.",
        nodeType: "missing_value_handler",
      },
      {
        id: "titanic-3",
        title: "Encode the Manifest",
        description:
          "Convert text columns like Sex and Embarked into numbers so the model can process them.",
        hint: "Use an Encoding node with label encoding or one-hot encoding.",
        nodeType: "encoding",
      },
      {
        id: "titanic-4",
        title: "Train the Predictor",
        description:
          "Split the data and train a Logistic Regression model to predict survival. Which features matter most?",
        hint: "Add Split → Logistic Regression nodes.",
        nodeType: "split",
      },
      {
        id: "titanic-5",
        title: "Who Survived?",
        description:
          "Evaluate your model. Check the confusion matrix — does your model better predict survivors or non-survivors?",
        hint: "Add a Confusion Matrix to see prediction details.",
        nodeType: "confusion_matrix",
      },
    ],
    challenge: {
      description:
        "Build a Titanic survival predictor with at least 78% accuracy. Handle missing values and encode categorical features.",
      metric: "Accuracy",
      threshold: "78%",
    },
  },

  diabetes: {
    datasetId: "diabetes",
    title: "Predicting Diabetes Progression",
    emoji: "🩺",
    color: "#10B981",
    narrative:
      "You're a medical researcher with data from 442 diabetes patients. Using measurements like BMI, blood pressure, and blood serum levels, predict how the disease will progress over one year. Your model could help doctors plan treatments.",
    characterContext:
      "Dr. Metric, a data-driven physician, explains each biomarker.",
    steps: [
      {
        id: "diabetes-1",
        title: "Review Patient Records",
        description:
          "Load the diabetes dataset and examine the statistics. Notice that features are already normalized — this is pre-processed clinical data.",
        hint: "Use Sample Dataset → Statistics View to understand distributions.",
        nodeType: "sample_dataset",
      },
      {
        id: "diabetes-2",
        title: "Select Key Biomarkers",
        description:
          "Not all 10 features may be useful. Use Feature Selection to find the most important predictors of disease progression.",
        hint: "Add a Feature Selection node to rank features by importance.",
        nodeType: "feature_selection",
      },
      {
        id: "diabetes-3",
        title: "Split Clinical Trials",
        description:
          "Divide patients into training and test groups. Use 75/25 split to keep enough test patients for reliable evaluation.",
        hint: "Add a Split node with a 75/25 ratio.",
        nodeType: "split",
      },
      {
        id: "diabetes-4",
        title: "Build a Regression Model",
        description:
          "Since we're predicting a continuous value (disease progression), use Linear Regression instead of classification.",
        hint: "Add a Linear Regression node connected to the training data.",
        nodeType: "linear_regression",
      },
      {
        id: "diabetes-5",
        title: "Measure Prediction Quality",
        description:
          "Use R² Score to see how well your model explains the variation in disease progression. A score above 0.4 is good for this challenging dataset.",
        hint: "Add an R² Score node to evaluate the model.",
        nodeType: "r2_score",
      },
    ],
    challenge: {
      description:
        "Achieve an R² score of at least 0.40 on the diabetes test set using Linear Regression.",
      metric: "R² Score",
      threshold: "0.40",
    },
  },

  breast_cancer: {
    datasetId: "breast_cancer",
    title: "Cancer Detection Assistant",
    emoji: "🔬",
    color: "#EC4899",
    narrative:
      "You're helping pathologists distinguish benign tumors from malignant ones using cell nucleus measurements from biopsy images. Your ML model could assist in early cancer detection, potentially saving lives.",
    characterContext:
      "Dr. Cell, a pathologist, explains what each measurement means for diagnosis.",
    steps: [
      {
        id: "cancer-1",
        title: "Examine the Biopsy Data",
        description:
          "Load the breast cancer dataset. It has 30 features computed from cell nucleus images. Preview the data and check for class imbalance.",
        hint: "Use Sample Dataset → Statistics View to see feature distributions.",
        nodeType: "sample_dataset",
      },
      {
        id: "cancer-2",
        title: "Scale the Features",
        description:
          "The features have very different scales (some are 0-1, others go to 2500). Normalize them so no single feature dominates.",
        hint: "Add a Scaling node with StandardScaler or MinMaxScaler.",
        nodeType: "scaling",
      },
      {
        id: "cancer-3",
        title: "Train the Detector",
        description:
          "Split and train a Logistic Regression classifier. For medical applications, minimizing false negatives (missed cancers) is critical.",
        hint: "Add Split → Logistic Regression nodes.",
        nodeType: "split",
      },
      {
        id: "cancer-4",
        title: "Analyze Predictions",
        description:
          "Check the confusion matrix carefully. How many malignant cases did the model miss (false negatives)? In medicine, these are the most dangerous errors.",
        hint: "Add a Confusion Matrix node and focus on the false negative count.",
        nodeType: "confusion_matrix",
      },
    ],
    challenge: {
      description:
        "Build a breast cancer classifier with at least 95% accuracy and fewer than 3 false negatives on the test set.",
      metric: "Accuracy",
      threshold: "95%",
    },
  },

  heart_disease: {
    datasetId: "heart_disease",
    title: "Heart Health Predictor",
    emoji: "❤️",
    color: "#EF4444",
    narrative:
      "Cardiovascular disease is the leading cause of death worldwide. Using patient data — age, cholesterol, blood pressure, chest pain type — build a model to flag high-risk patients for early intervention.",
    characterContext:
      "Dr. Heart, a cardiologist, walks you through the risk factors.",
    steps: [
      {
        id: "heart-1",
        title: "Patient Intake",
        description:
          "Load the heart disease dataset and explore the 13 clinical features. Notice both numerical (age, cholesterol) and categorical (chest pain type) features.",
        hint: "Use Sample Dataset → Column Info to understand feature types.",
        nodeType: "sample_dataset",
      },
      {
        id: "heart-2",
        title: "Handle Mixed Data",
        description:
          "Encode categorical columns and scale numerical features to prepare the data for modeling.",
        hint: "Add Encoding then Scaling nodes in sequence.",
        nodeType: "encoding",
      },
      {
        id: "heart-3",
        title: "Train & Predict",
        description:
          "Split the data and train a classifier. Compare how different configurations affect prediction quality.",
        hint: "Add Split → Logistic Regression.",
        nodeType: "split",
      },
      {
        id: "heart-4",
        title: "Evaluate Risk Model",
        description:
          "Check both the confusion matrix and overall accuracy. For a screening tool, high sensitivity (catching all positive cases) matters most.",
        hint: "Add Confusion Matrix to see the full prediction breakdown.",
        nodeType: "confusion_matrix",
      },
    ],
    challenge: {
      description:
        "Build a heart disease predictor with at least 80% accuracy on the test set.",
      metric: "Accuracy",
      threshold: "80%",
    },
  },
};

/** Check if a dataset has an available story */
export function hasStory(datasetId: string): boolean {
  return datasetId in datasetStories;
}

/** Get story for a dataset, or undefined */
export function getStory(datasetId: string): DatasetStory | undefined {
  return datasetStories[datasetId];
}
