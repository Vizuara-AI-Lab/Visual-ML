/**
 * Node-to-Python code template map.
 *
 * Each supported NodeType maps to a function that returns a CodeBlock
 * containing the imports, code body, and optional markdown comment
 * for that node's contribution to the generated pipeline.
 */

import type { NodeType } from "../../types/pipeline";
import type { CodeBlock, TemplateFunction } from "./types";

// ─── Helper utilities ────────────────────────────────────────────

function pyBool(value: unknown): string {
  return value === false ? "False" : "True";
}

function pyNone(value: unknown): string {
  return value == null ? "None" : String(value);
}

function pyStr(value: unknown): string {
  if (value == null || value === "") return '""';
  return `"${String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

// ─── Data Source Templates ───────────────────────────────────────

const uploadFile: TemplateFunction = (config, outVar) => ({
  imports: ["import pandas as pd"],
  code: `# TODO: Update the file path to your local CSV file\n${outVar} = pd.read_csv(${pyStr(config.filename || "dataset.csv")})\nprint(f"Dataset loaded: {${outVar}.shape[0]} rows, {${outVar}.shape[1]} columns")`,
  comment: "## Load Dataset",
});

const selectDataset: TemplateFunction = (config, outVar) => ({
  imports: ["import pandas as pd"],
  code: `# TODO: Update the file path to your local CSV file\n${outVar} = pd.read_csv(${pyStr(config.filename || "dataset.csv")})\nprint(f"Dataset loaded: {${outVar}.shape[0]} rows, {${outVar}.shape[1]} columns")`,
  comment: "## Load Dataset",
});

const sampleDataset: TemplateFunction = (config, outVar) => {
  const name = String(config.dataset_name || "iris");
  const loaderMap: Record<string, { fn: string; imp: string }> = {
    iris: { fn: "load_iris", imp: "from sklearn.datasets import load_iris" },
    boston: {
      fn: "fetch_california_housing",
      imp: "from sklearn.datasets import fetch_california_housing",
    },
    wine: { fn: "load_wine", imp: "from sklearn.datasets import load_wine" },
    diabetes: {
      fn: "load_diabetes",
      imp: "from sklearn.datasets import load_diabetes",
    },
  };
  const loader = loaderMap[name] || loaderMap.iris;

  return {
    imports: ["import pandas as pd", loader.imp],
    code: `_data = ${loader.fn}()\n${outVar} = pd.DataFrame(_data.data, columns=_data.feature_names)\n${outVar}["target"] = _data.target\nprint(f"Dataset loaded: {${outVar}.shape[0]} rows, {${outVar}.shape[1]} columns")`,
    comment: `## Load Sample Dataset (${name})`,
  };
};

// ─── Preprocessing Templates ────────────────────────────────────

const missingValueHandler: TemplateFunction = (config, outVar, inVar) => {
  const src = inVar || "df";
  const lines: string[] = [`${outVar} = ${src}.copy()`];

  const columnConfigs = config.column_configs as
    | Record<string, Record<string, unknown>>
    | undefined;
  const defaultStrategy = String(config.default_strategy || "none");

  if (columnConfigs && Object.keys(columnConfigs).length > 0) {
    for (const [col, colCfg] of Object.entries(columnConfigs)) {
      const strategy = String(colCfg.strategy || defaultStrategy);
      lines.push(...missingValueLines(outVar, col, strategy, colCfg));
    }
  } else if (defaultStrategy !== "none") {
    lines.push(
      `# Apply default strategy "${defaultStrategy}" to all columns with missing values`,
    );
    if (defaultStrategy === "drop") {
      lines.push(`${outVar} = ${outVar}.dropna()`);
    } else if (defaultStrategy === "mean") {
      lines.push(
        `${outVar} = ${outVar}.fillna(${outVar}.select_dtypes(include="number").mean())`,
      );
    } else if (defaultStrategy === "median") {
      lines.push(
        `${outVar} = ${outVar}.fillna(${outVar}.select_dtypes(include="number").median())`,
      );
    } else if (defaultStrategy === "mode") {
      lines.push(`${outVar} = ${outVar}.fillna(${outVar}.mode().iloc[0])`);
    } else if (defaultStrategy === "forward_fill") {
      lines.push(`${outVar} = ${outVar}.ffill()`);
    } else if (defaultStrategy === "backward_fill") {
      lines.push(`${outVar} = ${outVar}.bfill()`);
    }
  }

  lines.push(
    `print(f"Missing values remaining: {${outVar}.isnull().sum().sum()}")`,
  );

  return {
    imports: [],
    code: lines.join("\n"),
    comment: "## Handle Missing Values",
  };
};

function missingValueLines(
  outVar: string,
  col: string,
  strategy: string,
  colCfg: Record<string, unknown>,
): string[] {
  const lines: string[] = [];
  switch (strategy) {
    case "drop":
      lines.push(`${outVar} = ${outVar}.dropna(subset=["${col}"])`);
      break;
    case "mean":
      lines.push(
        `${outVar}["${col}"] = ${outVar}["${col}"].fillna(${outVar}["${col}"].mean())`,
      );
      break;
    case "median":
      lines.push(
        `${outVar}["${col}"] = ${outVar}["${col}"].fillna(${outVar}["${col}"].median())`,
      );
      break;
    case "mode":
      lines.push(
        `${outVar}["${col}"] = ${outVar}["${col}"].fillna(${outVar}["${col}"].mode()[0])`,
      );
      break;
    case "fill":
      lines.push(
        `${outVar}["${col}"] = ${outVar}["${col}"].fillna(${pyStr(colCfg.fill_value ?? 0)})`,
      );
      break;
    case "forward_fill":
      lines.push(
        `${outVar}["${col}"] = ${outVar}["${col}"].ffill()`,
      );
      break;
    case "backward_fill":
      lines.push(
        `${outVar}["${col}"] = ${outVar}["${col}"].bfill()`,
      );
      break;
    default:
      // "none" — skip
      break;
  }
  return lines;
}

// ─── Feature Engineering Templates ──────────────────────────────

const encoding: TemplateFunction = (config, outVar, inVar) => {
  const src = inVar || "df";
  const lines: string[] = [`${outVar} = ${src}.copy()`];
  const imports: string[] = [];

  const columnConfigs = config.column_configs as
    | Record<string, Record<string, unknown>>
    | undefined;

  let needsLabelEncoder = false;
  let needsGetDummies = false;

  if (columnConfigs && Object.keys(columnConfigs).length > 0) {
    for (const [col, colCfg] of Object.entries(columnConfigs)) {
      const method = String(colCfg.encoding_method || "none");
      if (method === "onehot") {
        needsGetDummies = true;
        const dropFirst = colCfg.drop_first ? ", drop_first=True" : "";
        lines.push(
          `${outVar} = pd.get_dummies(${outVar}, columns=["${col}"]${dropFirst})`,
        );
      } else if (method === "label") {
        needsLabelEncoder = true;
        lines.push(
          `_le_${col} = LabelEncoder()`,
          `${outVar}["${col}"] = _le_${col}.fit_transform(${outVar}["${col}"].astype(str))`,
        );
      }
    }
  }

  if (needsLabelEncoder) {
    imports.push("from sklearn.preprocessing import LabelEncoder");
  }
  // pd.get_dummies is in pandas — no extra import
  if (needsGetDummies) {
    // pandas already imported
  }

  lines.push(
    `print(f"Encoded dataset shape: {${outVar}.shape}")`,
  );

  return {
    imports,
    code: lines.join("\n"),
    comment: "## Encode Categorical Variables",
  };
};

const scaling: TemplateFunction = (config, outVar, inVar) => {
  const src = inVar || "df";
  const method = String(config.method || "standard");

  const scalerMap: Record<string, { cls: string; imp: string }> = {
    standard: {
      cls: "StandardScaler",
      imp: "from sklearn.preprocessing import StandardScaler",
    },
    minmax: {
      cls: "MinMaxScaler",
      imp: "from sklearn.preprocessing import MinMaxScaler",
    },
    robust: {
      cls: "RobustScaler",
      imp: "from sklearn.preprocessing import RobustScaler",
    },
    normalize: {
      cls: "Normalizer",
      imp: "from sklearn.preprocessing import Normalizer",
    },
  };

  const scaler = scalerMap[method] || scalerMap.standard;

  const columns = config.columns as string[] | undefined;
  let code: string;

  if (columns && columns.length > 0) {
    const colList = columns.map((c) => `"${c}"`).join(", ");
    code = `_scaler = ${scaler.cls}()\n_cols_to_scale = [${colList}]\n${outVar} = ${src}.copy()\n${outVar}[_cols_to_scale] = _scaler.fit_transform(${outVar}[_cols_to_scale])`;
  } else {
    code = `_scaler = ${scaler.cls}()\n_numeric_cols = ${src}.select_dtypes(include="number").columns.tolist()\n${outVar} = ${src}.copy()\n${outVar}[_numeric_cols] = _scaler.fit_transform(${outVar}[_numeric_cols])`;
  }

  return {
    imports: [scaler.imp],
    code,
    comment: `## Feature Scaling (${scaler.cls})`,
  };
};

const featureSelection: TemplateFunction = (config, outVar, inVar) => {
  const src = inVar || "df";
  const method = String(config.method || "variance");
  const imports: string[] = [];
  let code: string;

  if (method === "variance") {
    const threshold = Number(config.variance_threshold ?? 0);
    imports.push(
      "from sklearn.feature_selection import VarianceThreshold",
    );
    code = `_selector = VarianceThreshold(threshold=${threshold})\n_numeric = ${src}.select_dtypes(include="number")\n_selected = _selector.fit_transform(_numeric)\n${outVar} = pd.DataFrame(_selected, columns=_numeric.columns[_selector.get_support()], index=${src}.index)\n# Re-attach non-numeric columns\nfor col in ${src}.select_dtypes(exclude="number").columns:\n    ${outVar}[col] = ${src}[col].values`;
  } else {
    // correlation
    const threshold = Number(config.correlation_threshold ?? 0.95);
    code = `_corr_matrix = ${src}.select_dtypes(include="number").corr().abs()\n_upper = _corr_matrix.where(pd.np.triu(pd.np.ones(_corr_matrix.shape), k=1).astype(bool))\n_to_drop = [col for col in _upper.columns if any(_upper[col] > ${threshold})]\n${outVar} = ${src}.drop(columns=_to_drop)\nprint(f"Dropped {len(_to_drop)} highly correlated features: {_to_drop}")`;
    imports.push("import numpy as np");
  }

  return {
    imports,
    code,
    comment: "## Feature Selection",
  };
};

// ─── Split Template ─────────────────────────────────────────────

const split: TemplateFunction = (config) => {
  const target = String(config.target_column || "target");
  const testSize = Number(config.test_ratio ?? 0.2);
  const seed = Number(config.random_seed ?? 42);
  const shuffle = config.shuffle !== false;
  const stratified = config.split_type === "stratified";

  let code = `X = df_pipeline.drop(columns=["${target}"])\ny = df_pipeline["${target}"]\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=${testSize}, random_state=${seed}, shuffle=${pyBool(shuffle)}`;

  if (stratified) {
    code += `, stratify=y`;
  }
  code += `\n)`;
  code += `\nprint(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")`;

  return {
    imports: ["from sklearn.model_selection import train_test_split"],
    code,
    comment: "## Train/Test Split",
  };
};

// ─── ML Algorithm Templates ─────────────────────────────────────

const linearRegression: TemplateFunction = (config, outVar) => {
  const fitIntercept = config.fit_intercept !== false;

  return {
    imports: ["from sklearn.linear_model import LinearRegression"],
    code: `${outVar} = LinearRegression(fit_intercept=${pyBool(fitIntercept)})\n${outVar}.fit(X_train, y_train)\ny_pred = ${outVar}.predict(X_test)\nprint(f"Model trained with {X_train.shape[1]} features")`,
    comment: "## Train Linear Regression Model",
  };
};

const logisticRegression: TemplateFunction = (config, outVar) => {
  const C = Number(config.C ?? 1.0);
  const maxIter = Number(config.max_iter ?? 1000);
  const penalty = String(config.penalty || "l2");
  const solver = String(config.solver || "lbfgs");
  const seed = Number(config.random_state ?? 42);
  const fitIntercept = config.fit_intercept !== false;

  const penaltyParam = penalty === "none" ? "None" : `"${penalty}"`;

  return {
    imports: ["from sklearn.linear_model import LogisticRegression"],
    code: `${outVar} = LogisticRegression(\n    C=${C}, penalty=${penaltyParam}, solver="${solver}",\n    max_iter=${maxIter}, random_state=${seed},\n    fit_intercept=${pyBool(fitIntercept)}\n)\n${outVar}.fit(X_train, y_train)\ny_pred = ${outVar}.predict(X_test)\nprint(f"Model trained with {X_train.shape[1]} features")`,
    comment: "## Train Logistic Regression Model",
  };
};

const decisionTree: TemplateFunction = (config, outVar) => {
  const taskType = String(config.task_type || "classification");
  const maxDepth = pyNone(config.max_depth);
  const minSplit = Number(config.min_samples_split ?? 2);
  const minLeaf = Number(config.min_samples_leaf ?? 1);
  const seed = Number(config.random_state ?? 42);

  const isClassification = taskType === "classification";
  const cls = isClassification
    ? "DecisionTreeClassifier"
    : "DecisionTreeRegressor";
  const imp = isClassification
    ? "from sklearn.tree import DecisionTreeClassifier"
    : "from sklearn.tree import DecisionTreeRegressor";

  return {
    imports: [imp],
    code: `${outVar} = ${cls}(\n    max_depth=${maxDepth}, min_samples_split=${minSplit},\n    min_samples_leaf=${minLeaf}, random_state=${seed}\n)\n${outVar}.fit(X_train, y_train)\ny_pred = ${outVar}.predict(X_test)\nprint(f"Model trained with {X_train.shape[1]} features")`,
    comment: `## Train Decision Tree (${taskType})`,
  };
};

const randomForest: TemplateFunction = (config, outVar) => {
  const taskType = String(config.task_type || "classification");
  const nEstimators = Number(config.n_estimators ?? 100);
  const maxDepth = pyNone(config.max_depth);
  const minSplit = Number(config.min_samples_split ?? 2);
  const minLeaf = Number(config.min_samples_leaf ?? 1);
  const seed = Number(config.random_state ?? 42);

  const isClassification = taskType === "classification";
  const cls = isClassification
    ? "RandomForestClassifier"
    : "RandomForestRegressor";
  const imp = isClassification
    ? "from sklearn.ensemble import RandomForestClassifier"
    : "from sklearn.ensemble import RandomForestRegressor";

  return {
    imports: [imp],
    code: `${outVar} = ${cls}(\n    n_estimators=${nEstimators}, max_depth=${maxDepth},\n    min_samples_split=${minSplit}, min_samples_leaf=${minLeaf},\n    random_state=${seed}\n)\n${outVar}.fit(X_train, y_train)\ny_pred = ${outVar}.predict(X_test)\nprint(f"Model trained with {X_train.shape[1]} features")`,
    comment: `## Train Random Forest (${taskType})`,
  };
};

// ─── Metrics Templates ──────────────────────────────────────────

const r2Score: TemplateFunction = () => ({
  imports: ["from sklearn.metrics import r2_score"],
  code: `r2 = r2_score(y_test, y_pred)\nprint(f"R² Score: {r2:.4f}")`,
  comment: "## R² Score",
});

const mseScore: TemplateFunction = () => ({
  imports: ["from sklearn.metrics import mean_squared_error"],
  code: `mse = mean_squared_error(y_test, y_pred)\nprint(f"MSE: {mse:.4f}")`,
  comment: "## Mean Squared Error",
});

const rmseScore: TemplateFunction = () => ({
  imports: [
    "import numpy as np",
    "from sklearn.metrics import mean_squared_error",
  ],
  code: `rmse = np.sqrt(mean_squared_error(y_test, y_pred))\nprint(f"RMSE: {rmse:.4f}")`,
  comment: "## Root Mean Squared Error",
});

const maeScore: TemplateFunction = () => ({
  imports: ["from sklearn.metrics import mean_absolute_error"],
  code: `mae = mean_absolute_error(y_test, y_pred)\nprint(f"MAE: {mae:.4f}")`,
  comment: "## Mean Absolute Error",
});

const confusionMatrix: TemplateFunction = () => ({
  imports: [
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay",
    "import matplotlib.pyplot as plt",
  ],
  code: `cm = confusion_matrix(y_test, y_pred)\ndisp = ConfusionMatrixDisplay(confusion_matrix=cm)\ndisp.plot()\nplt.title("Confusion Matrix")\nplt.tight_layout()\nplt.show()`,
  comment: "## Confusion Matrix",
});

// ─── View / Utility — emit no code ─────────────────────────────

const noopTemplate: TemplateFunction = () => ({
  imports: [],
  code: "",
  comment: "",
});

// ─── Master Template Map ────────────────────────────────────────

export const NODE_TEMPLATES: Partial<Record<NodeType, TemplateFunction>> = {
  // Data sources
  upload_file: uploadFile,
  select_dataset: selectDataset,
  sample_dataset: sampleDataset,

  // Preprocessing
  missing_value_handler: missingValueHandler,
  preprocess: missingValueHandler, // alias

  // Feature engineering
  encoding,
  scaling,
  feature_selection: featureSelection,

  // Split
  split,

  // ML algorithms
  linear_regression: linearRegression,
  logistic_regression: logisticRegression,
  decision_tree: decisionTree,
  random_forest: randomForest,

  // Metrics
  r2_score: r2Score,
  mse_score: mseScore,
  rmse_score: rmseScore,
  mae_score: maeScore,
  confusion_matrix: confusionMatrix,

  // View nodes — skipped in code generation
  table_view: noopTemplate,
  data_preview: noopTemplate,
  statistics_view: noopTemplate,
  column_info: noopTemplate,
  chart_view: noopTemplate,

  // GenAI / deployment — not exportable
  llm_node: noopTemplate,
  system_prompt: noopTemplate,
  chatbot_node: noopTemplate,
  example_node: noopTemplate,
  model_export: noopTemplate,
  api_endpoint: noopTemplate,

  // Transformation (currently commented out in node defs)
  transformation: noopTemplate,
};

/** Node types that should be skipped (produce no code) */
export const SKIP_NODE_TYPES = new Set<string>([
  "table_view",
  "data_preview",
  "statistics_view",
  "column_info",
  "chart_view",
  "llm_node",
  "system_prompt",
  "chatbot_node",
  "example_node",
  "model_export",
  "api_endpoint",
  "transformation",
]);
