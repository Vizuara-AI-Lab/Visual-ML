/**
 * Learning Flow Messages
 *
 * All guidance messages for the step-by-step learning flow.
 * Every message is speech-friendly plain text. No markdown,
 * no asterisks, no emojis, no special characters.
 */

import { getAlgorithmConfig } from "./algorithmConfig";

export interface FlowMessage {
  title: string;
  displayText: string;
  voiceText: string;
}

// ── Helpers ──────────────────────────────────────────────────────

function getTimeGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

function listColumns(columns: string[], max = 3): string {
  if (columns.length === 0) return "";
  const shown = columns.slice(0, max);
  const extra =
    columns.length > max ? ` and ${columns.length - max} more` : "";
  return shown.join(", ") + extra;
}

// ── Message Factories ────────────────────────────────────────────

export function welcomeMessage(userName: string): FlowMessage {
  const greeting = getTimeGreeting();
  const text =
    `${greeting}, ${userName}. I am your AI mentor. ` +
    `I will guide you step by step to build a machine learning model. ` +
    `What would you like to learn today? Choose one of the options below.`;
  return { title: "Welcome", displayText: text, voiceText: text };
}

export function algorithmSelectedMessage(algorithm: string): FlowMessage {
  const config = getAlgorithmConfig(algorithm);

  const intros: Record<string, string> = {
    linear_regression:
      `Great choice. Linear Regression is used to predict numbers ` +
      `like house prices, test scores, or temperatures. ` +
      `It finds the best fitting straight line through your data points. ` +
      `Let us build one together, step by step.`,

    logistic_regression:
      `Great choice. Logistic Regression is used to predict categories ` +
      `like whether an email is spam or not, or whether a patient has a disease. ` +
      `It calculates the probability of each category and picks the most likely one. ` +
      `Let us build one together, step by step.`,

    decision_tree:
      `Great choice. A Decision Tree makes decisions like a flowchart. ` +
      `It asks a series of yes or no questions about your data to reach a prediction. ` +
      `Think of it like playing a game of 20 questions. ` +
      `Let us build one together, step by step.`,

    random_forest:
      `Great choice. A Random Forest builds many decision trees and ` +
      `combines their answers by taking a vote. ` +
      `It is like asking 100 experts instead of just one, so the answer is more reliable. ` +
      `Let us build one together, step by step.`,

    mlp_classifier:
      `Great choice. An MLP Classifier is a neural network made of layers ` +
      `of interconnected nodes. It can learn complex patterns that simpler models might miss. ` +
      `Think of it as a team of neurons working together to classify your data. ` +
      `Let us build one together, step by step.`,

    mlp_regressor:
      `Great choice. An MLP Regressor is a neural network that predicts numbers ` +
      `like prices, scores, or temperatures. ` +
      `Unlike Linear Regression which only finds straight lines, ` +
      `an MLP can learn curved and complex relationships in your data. ` +
      `Let us build one together, step by step.`,

    kmeans:
      `Great choice. K-Means Clustering is different from the other algorithms ` +
      `because it does not need labels. ` +
      `It automatically groups similar data points together into clusters. ` +
      `Think of it like sorting a pile of mixed fruits by similarity ` +
      `without knowing their names. Let us build one together, step by step.`,

    image_predictions:
      `Great choice. Image Classification teaches a computer to recognize ` +
      `what is in a picture, like telling apart cats from dogs, ` +
      `or handwritten digits from zero to nine. ` +
      `We will load an image dataset, split it, and train a model. ` +
      `Let us build one together, step by step.`,
  };

  const text = intros[algorithm] ?? `Let us build a ${config.displayName} model together.`;
  return {
    title: config.displayName,
    displayText: text,
    voiceText: text,
  };
}

export function promptDragDatasetMessage(): FlowMessage {
  const text =
    `First, we need data to work with. ` +
    `Look at the left sidebar under Data Source. ` +
    `Drag a Select Dataset node or an Upload File node onto the canvas.`;
  return { title: "Add a Dataset", displayText: text, voiceText: text };
}

export function datasetNodeAddedMessage(): FlowMessage {
  const text =
    `I can see you added a dataset node. ` +
    `Now click on it to open its settings and choose your dataset.`;
  return {
    title: "Configure Your Dataset",
    displayText: text,
    voiceText: text,
  };
}

export function datasetConfiguredMessage(
  filename: string,
  nRows: number,
  nCols: number,
): FlowMessage {
  const text =
    `Your dataset ${filename} is loaded with ${nRows} rows and ${nCols} columns. ` +
    `Now let us understand your data. ` +
    `Drag a Column Info node from the View section on the left ` +
    `and connect it to your dataset node.`;
  return {
    title: "Dataset Loaded",
    displayText: text,
    voiceText: text,
  };
}

export function promptColumnInfoMessage(): FlowMessage {
  const text =
    `Drag a Column Info node from the View section on the left sidebar ` +
    `and connect it to your dataset node.`;
  return {
    title: "Add Column Info",
    displayText: text,
    voiceText: text,
  };
}

export function columnInfoAddedMessage(): FlowMessage {
  const text =
    `Column Info node is connected. ` +
    `Now click the Run Pipeline button in the toolbar ` +
    `to see the details of each column in your data.`;
  return {
    title: "Run to See Column Details",
    displayText: text,
    voiceText: text,
  };
}

export function promptRunColumnInfoMessage(): FlowMessage {
  const text =
    `Click the Run Pipeline button in the toolbar to analyze your columns.`;
  return {
    title: "Run Pipeline",
    displayText: text,
    voiceText: text,
  };
}

export function columnInfoExecutedMissingMessage(
  missingColumns: Array<{ column: string; count: number }>,
  totalMissing: number,
): FlowMessage {
  const colNames = listColumns(missingColumns.map((c) => c.column));
  const text =
    `Your column info results are in. ` +
    `I found ${totalMissing} missing values in columns: ${colNames}. ` +
    `Missing values are empty cells that can confuse the model. ` +
    `Drag a Missing Value Handler node from the Preprocessing section ` +
    `and connect it after your dataset. ` +
    `You can fill missing values with the mean, median, or mode, ` +
    `or drop the rows entirely.`;
  return {
    title: "Missing Values Found",
    displayText: text,
    voiceText: text,
  };
}

export function columnInfoExecutedCategoricalMessage(
  categoricalColumns: string[],
): FlowMessage {
  const colNames = listColumns(categoricalColumns);
  const text =
    `Your data has ${categoricalColumns.length} text columns: ${colNames}. ` +
    `Machine learning models only understand numbers, ` +
    `so we need to convert this text into numbers. ` +
    `Drag an Encoding node from the Preprocessing section on the left.`;
  return {
    title: "Text Columns Found",
    displayText: text,
    voiceText: text,
  };
}

export function columnInfoExecutedCleanMessage(): FlowMessage {
  const text =
    `Your data looks great. No missing values and all numeric columns. ` +
    `Now let us split the data. ` +
    `We keep 80 percent for training the model ` +
    `and 20 percent for testing how well it learned. ` +
    `Drag a Target and Split node from the Preprocessing section.`;
  return {
    title: "Data Looks Clean",
    displayText: text,
    voiceText: text,
  };
}

export function missingValuesAddedMessage(): FlowMessage {
  const text =
    `Missing Value Handler is added. ` +
    `Click on it and choose a fill strategy. ` +
    `Mean works well for number columns and mode works well for text categories.`;
  return {
    title: "Configure Missing Value Handler",
    displayText: text,
    voiceText: text,
  };
}

export function promptEncodingAfterMissingMessage(
  categoricalColumns: string[],
): FlowMessage {
  const colNames = listColumns(categoricalColumns);
  const text =
    `Missing values are handled. ` +
    `Now let us convert text columns to numbers. ` +
    `You have ${categoricalColumns.length} text columns: ${colNames}. ` +
    `Drag an Encoding node and connect it after the Missing Value Handler.`;
  return {
    title: "Add Encoding",
    displayText: text,
    voiceText: text,
  };
}

export function encodingAddedMessage(): FlowMessage {
  const text =
    `Encoding node is added. ` +
    `Now let us split your data into training and testing sets. ` +
    `Drag a Target and Split node from the Preprocessing section.`;
  return {
    title: "Add Train Test Split",
    displayText: text,
    voiceText: text,
  };
}

export function promptSplitMessage(): FlowMessage {
  const text =
    `Now we need to split your data. ` +
    `We keep 80 percent for training, which teaches the model, ` +
    `and 20 percent for testing, which checks if it learned well. ` +
    `Drag a Target and Split node and connect it in the pipeline.`;
  return {
    title: "Split Your Data",
    displayText: text,
    voiceText: text,
  };
}

export function splitNodeAddedMessage(): FlowMessage {
  const text =
    `Target and Split node is added. ` +
    `Click on it to open its settings. ` +
    `Choose which column you want to predict as the target column, ` +
    `then set the train test split ratio. ` +
    `80 percent training and 20 percent testing is a good default.`;
  return {
    title: "Configure Target and Split",
    displayText: text,
    voiceText: text,
  };
}

export function splitConfiguredMessage(algorithm: string): FlowMessage {
  const config = getAlgorithmConfig(algorithm);
  const text =
    `Data split is configured. ` +
    `Now drag a ${config.displayName} node from the Models section on the left ` +
    `and connect it to the Split node.`;
  return {
    title: `Add ${config.displayName}`,
    displayText: text,
    voiceText: text,
  };
}

export function modelAddedMessage(algorithm: string): FlowMessage {
  const config = getAlgorithmConfig(algorithm);
  const text =
    `${config.displayName} node is added. Almost there. ` +
    `Now let us add metrics to see how well the model performs. ` +
    `Drag ${config.metricDisplayText} nodes from the Metrics section ` +
    `and connect them to the model node.`;
  return {
    title: "Add Metrics",
    displayText: text,
    voiceText:
      `${config.displayName} node is added. Almost there. ` +
      `Now let us add metrics to see how well the model performs. ` +
      `Drag ${config.metricVoiceText} nodes from the Metrics section ` +
      `and connect them to the model node.`,
  };
}

export function metricsAddedMessage(): FlowMessage {
  const text =
    `Metrics are connected. Your pipeline is complete. ` +
    `Make sure all nodes are connected in order, ` +
    `then click the Run Pipeline button to train your model and see results.`;
  return {
    title: "Pipeline Complete",
    displayText: text,
    voiceText: text,
  };
}

export function promptFinalRunMessage(): FlowMessage {
  const text = `Everything is connected. Click Run Pipeline in the toolbar now.`;
  return {
    title: "Run Your Pipeline",
    displayText: text,
    voiceText: text,
  };
}

export function pipelineExecutedMessage(
  algorithm: string,
  results: Record<string, number>,
): FlowMessage {
  const config = getAlgorithmConfig(algorithm);
  let text: string;

  if (config.isUnsupervised) {
    // K-Means clustering results
    const nClusters = results.n_clusters;
    const silhouette = results.silhouette_score;
    if (nClusters !== undefined) {
      text =
        `Your K-Means model has grouped the data into ${nClusters} clusters. `;
      if (silhouette !== undefined) {
        const quality = silhouette > 0.5 ? "good" : silhouette > 0.25 ? "moderate" : "low, try adjusting the number of clusters";
        text +=
          `The silhouette score is ${silhouette.toFixed(3)}, ` +
          `which measures how well separated the clusters are. ` +
          `Your score is ${quality}.`;
      }
    } else {
      text =
        `Your clustering model has finished. ` +
        `Click on the K-Means node to explore the cluster assignments.`;
    }
  } else if (config.isImagePipeline) {
    // Image classification results
    const accuracy = results.accuracy;
    if (accuracy !== undefined) {
      const pct = (accuracy * 100).toFixed(0);
      text =
        `Your image classifier has been trained and tested. ` +
        `The accuracy is ${pct} percent. ` +
        `Click on the Image Predictions node to explore the results, ` +
        `including the confusion matrix and training curves. ` +
        `You can even try the live camera tab to test the model in real time.`;
    } else {
      text =
        `Your image classifier has been trained. ` +
        `Click on the Image Predictions node to see the results.`;
    }
  } else if (config.isClassification) {
    const accuracy = results.accuracy;
    if (accuracy !== undefined) {
      const pct = (accuracy * 100).toFixed(0);
      text =
        `Your model has been trained and tested. ` +
        `The accuracy is ${pct} percent, which means the model ` +
        `correctly predicted ${pct} out of every 100 examples. ` +
        `Check the Confusion Matrix node to see exactly where the model ` +
        `got predictions right and where it made mistakes.`;
    } else {
      text =
        `Your model has been trained and tested. ` +
        `Click on the Confusion Matrix node to see how well it predicted each category.`;
    }
  } else {
    const r2 = results.r2_score ?? results.r2;
    const rmse = results.rmse_score ?? results.rmse;
    if (r2 !== undefined) {
      const pct = (r2 * 100).toFixed(0);
      text =
        `Your model has been trained and tested. ` +
        `The R squared score is ${r2.toFixed(3)}, which means the model ` +
        `explains ${pct} percent of the patterns in your data. ` +
        `A score closer to 1 is better.`;
      if (rmse !== undefined) {
        text += ` The root mean squared error is ${rmse.toFixed(2)}, ` +
          `which is the average prediction error.`;
      }
    } else {
      text =
        `Your model has been trained and tested. ` +
        `Click on the metric nodes to see the performance scores.`;
    }
  }

  return {
    title: "Results",
    displayText: text,
    voiceText: text,
  };
}

export function completedMessage(algorithm: string): FlowMessage {
  const config = getAlgorithmConfig(algorithm);
  const text =
    `Congratulations! You have successfully built a ` +
    `${config.displayName} model from start to finish. ` +
    `You can experiment with different settings, try a different dataset, ` +
    `or learn another algorithm. Would you like to start over with a different model?`;
  return {
    title: "Well Done",
    displayText: text,
    voiceText: text,
  };
}

export function errorOccurredMessage(errorMessage: string): FlowMessage {
  const text =
    `Something went wrong during execution. ` +
    `The error says: ${errorMessage}. ` +
    `Check that all nodes are connected properly and configured correctly, ` +
    `then try running the pipeline again.`;
  return {
    title: "Execution Error",
    displayText: text,
    voiceText: text,
  };
}

export function missingValueErrorMessage(errorMessage: string): FlowMessage {
  // Extract column name from error if present
  const columnMatch = errorMessage.match(/["']([^"']+)["']/);
  const columnHint = columnMatch
    ? ` The column ${columnMatch[1]} has empty cells.`
    : "";

  const text =
    `Your data has missing values that need to be handled before the model can run.` +
    columnHint +
    ` Drag a Missing Value Handler node from the Preprocessing section ` +
    `and connect it between your dataset and the rest of the pipeline. ` +
    `You can fill missing values with the mean, median, or mode, ` +
    `or drop the rows entirely.`;
  return {
    title: "Handle Missing Values",
    displayText: text,
    voiceText: text,
  };
}

export function encodingErrorMessage(errorMessage: string): FlowMessage {
  const text =
    `Your data has text columns that need to be converted to numbers ` +
    `before the model can run. ` +
    `Drag an Encoding node from the Preprocessing section ` +
    `and connect it in your pipeline before the model node. ` +
    `Label Encoding gives each category a number. ` +
    `One Hot Encoding creates separate columns for each category.`;
  return {
    title: "Encode Text Data",
    displayText: text,
    voiceText: text,
  };
}

export function logisticTargetWarningMessage(uniqueValues: number): FlowMessage {
  const text =
    `Warning. Your target column has ${uniqueValues} unique values. ` +
    `Logistic Regression works best with a small number of categories, ` +
    `typically 2 to 10. With too many unique values, ` +
    `the model may not perform well. ` +
    `Consider using a column with fewer categories as your target, ` +
    `or try Linear Regression instead for predicting numbers.`;
  return {
    title: "Too Many Categories",
    displayText: text,
    voiceText: text,
  };
}

// ── K-Means Clustering Messages ─────────────────────────────────

export function promptDragDatasetForClusteringMessage(): FlowMessage {
  const text =
    `First, we need data to work with. ` +
    `Look at the left sidebar under Data Source. ` +
    `Drag a Select Dataset or Upload File node onto the canvas. ` +
    `K-Means does not need a target column, so any numeric dataset works.`;
  return { title: "Add a Dataset", displayText: text, voiceText: text };
}

export function promptClusteringModelMessage(): FlowMessage {
  const text =
    `Your dataset is ready. ` +
    `Now drag a K-Means Clustering node from the ML Algorithms section ` +
    `and connect it to your dataset node. ` +
    `K-Means does not need a train test split because it is unsupervised.`;
  return {
    title: "Add K-Means Node",
    displayText: text,
    voiceText: text,
  };
}

export function clusteringModelAddedMessage(): FlowMessage {
  const text =
    `K-Means node is connected. ` +
    `Click on it to set the number of clusters. ` +
    `Start with 3 clusters and adjust based on your results. ` +
    `Then click Run Pipeline to see the clusters.`;
  return {
    title: "Configure and Run",
    displayText: text,
    voiceText: text,
  };
}

export function clusteringExecutedMessage(
  nClusters: number,
  silhouetteScore?: number,
): FlowMessage {
  let text =
    `Your K-Means model has grouped the data into ${nClusters} clusters. `;
  if (silhouetteScore !== undefined) {
    const pct = (silhouetteScore * 100).toFixed(0);
    text +=
      `The silhouette score is ${silhouetteScore.toFixed(3)}, ` +
      `which measures how well separated the clusters are. ` +
      `A score closer to 1 means the clusters are well defined. ` +
      `Your score of ${pct} percent is ${silhouetteScore > 0.5 ? "good" : silhouetteScore > 0.25 ? "moderate" : "low"}.`;
  } else {
    text +=
      `Click on the K-Means node to explore the cluster assignments ` +
      `and see which data points ended up in each group.`;
  }
  return {
    title: "Clustering Results",
    displayText: text,
    voiceText: text,
  };
}

// ── Image Pipeline Messages ─────────────────────────────────────

export function promptDragImageDatasetMessage(): FlowMessage {
  const text =
    `For image classification, we use a different set of nodes. ` +
    `Look at the left sidebar under Image Pipeline. ` +
    `Drag an Image Dataset node onto the canvas. ` +
    `You can choose a built-in dataset like MNIST or capture your own images with the camera.`;
  return { title: "Add Image Dataset", displayText: text, voiceText: text };
}

export function imageDatasetConfiguredMessage(
  datasetName: string,
  nImages: number,
  nClasses: number,
): FlowMessage {
  const text =
    `Your image dataset ${datasetName} is loaded with ${nImages} images ` +
    `across ${nClasses} classes. ` +
    `Now drag an Image Split node from the Image Pipeline section ` +
    `and connect it to your Image Dataset node. ` +
    `This will split the images into training and test sets.`;
  return {
    title: "Image Dataset Loaded",
    displayText: text,
    voiceText: text,
  };
}

export function promptImageSplitMessage(): FlowMessage {
  const text =
    `Drag an Image Split node from the Image Pipeline section ` +
    `and connect it to your Image Dataset. ` +
    `It will split images into 80 percent training and 20 percent testing ` +
    `while keeping each class balanced in both sets.`;
  return { title: "Add Image Split", displayText: text, voiceText: text };
}

export function imageSplitAddedMessage(): FlowMessage {
  const text =
    `Image Split is connected. ` +
    `Now drag an Image Predictions node from the Image Pipeline section. ` +
    `This node trains the model and evaluates it all in one step.`;
  return {
    title: "Add Image Predictions",
    displayText: text,
    voiceText: text,
  };
}

export function promptImagePredictionsMessage(): FlowMessage {
  const text =
    `Drag an Image Predictions node and connect it to the Image Split node. ` +
    `Then click Run Pipeline to train and evaluate your image classifier.`;
  return {
    title: "Add Image Predictions",
    displayText: text,
    voiceText: text,
  };
}

export function imageResultsMessage(accuracy: number): FlowMessage {
  const pct = (accuracy * 100).toFixed(0);
  const text =
    `Your image classifier has been trained and tested. ` +
    `The accuracy is ${pct} percent, which means the model correctly ` +
    `classified ${pct} out of every 100 test images. ` +
    `Click on the Image Predictions node to explore the results, ` +
    `including the confusion matrix, training curves, and feature importance. ` +
    `You can even try the live camera tab to test the model in real time.`;
  return {
    title: "Image Classification Results",
    displayText: text,
    voiceText: text,
  };
}
