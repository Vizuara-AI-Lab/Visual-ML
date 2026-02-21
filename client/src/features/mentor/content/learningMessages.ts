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

  if (config.isClassification) {
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
