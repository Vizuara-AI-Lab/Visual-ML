"""
Centralized error formatter for ML pipeline execution.

Converts raw Python/scikit-learn exceptions into student-friendly messages
with actionable suggestions. Every node error passes through format_error()
at the engine.py chokepoint.
"""

import re
from typing import Any, Dict, Optional, Tuple

# Each pattern: (compiled_regex, error_code, friendly_message_template, suggestion_template)
# Templates can use {match} for the full match, {g1}, {g2}, etc. for groups,
# and {node_label}, {node_type} for context.

ERROR_PATTERNS: list[Tuple[re.Pattern, str, str, str]] = [
    # ── Data Loading ──────────────────────────────────────────────────
    (
        re.compile(r"(?:FileNotFoundError|No such file|dataset.*not found)", re.I),
        "DATASET_NOT_FOUND",
        "The dataset file could not be found.",
        "Make sure you've uploaded a dataset or selected a built-in one in the Dataset node.",
    ),
    (
        re.compile(r"(?:Empty dataset|0 samples|no data|dataframe is empty)", re.I),
        "EMPTY_DATASET",
        "The dataset is empty — there are no rows to work with.",
        "Check that your dataset file has data rows (not just headers). Try uploading a different file.",
    ),
    (
        re.compile(r"(?:Permission denied|access denied|storage.*error)", re.I),
        "STORAGE_ERROR",
        "Could not access the data file on the server.",
        "This is a server issue. Try re-uploading your dataset or refreshing the page.",
    ),

    # ── Target / Column Validation ────────────────────────────────────
    (
        re.compile(r"target.*column['\s]+(\w+)['\s]+.*not found", re.I),
        "TARGET_NOT_FOUND",
        'The target column "{g1}" was not found in your data.',
        "Open the node settings and pick a target column that exists in your dataset.",
    ),
    (
        re.compile(r"target.*column.*not.*(?:set|configured|specified)", re.I),
        "TARGET_NOT_SET",
        "No target column has been selected.",
        "Open the model node settings and choose which column the model should predict.",
    ),
    (
        re.compile(r"(?:missing required|required field|required parameter).*['\"]([\w\s]+)['\"]", re.I),
        "MISSING_FIELD",
        'A required setting "{g1}" is missing.',
        "Open the node settings and fill in all required fields.",
    ),
    (
        re.compile(r"(?:split|test_size|train_size).*(?:out of range|between 0 and 1|invalid ratio)", re.I),
        "INVALID_SPLIT_RATIO",
        "The train/test split ratio is invalid.",
        "Set the split ratio between 0.1 and 0.9 (e.g., 0.8 means 80% training, 20% testing).",
    ),

    # ── Classification vs Regression Mismatch ────────────────────────
    (
        re.compile(r"target.*?['\"]([\w]+)['\"].*?(\d+) unique values.*too many for classification.*use (.+?) instead", re.I | re.S),
        "WRONG_TASK_TYPE",
        'The target column "{g1}" has {g2} unique values, which is too many for classification.',
        "This looks like a regression problem (continuous numbers), not classification. Use {g3} instead.",
    ),
    (
        re.compile(r"(\d+) unique values.*too many for classification", re.I),
        "WRONG_TASK_TYPE",
        "The target column has {g1} unique values, which is too many for classification.",
        "Classification is for categorical targets (e.g. 'cat'/'dog'). For numeric targets, use a Regressor node instead.",
    ),

    # ── Missing Values / NaN ──────────────────────────────────────────
    (
        re.compile(r"(?:Input contains NaN|missing values? found|NaN.*(?:in|found)|contains? NaN)", re.I),
        "MISSING_VALUES",
        "Your data contains empty or missing values (NaN).",
        "Add a Missing Values handler node before this step, or remove rows with missing data in your dataset.",
    ),

    # ── Encoding / Type Errors ────────────────────────────────────────
    (
        re.compile(r"could not convert string to float", re.I),
        "STRING_TO_FLOAT",
        "Some columns contain text instead of numbers.",
        "Add an Encoding node to convert text columns to numbers before training the model.",
    ),
    (
        re.compile(r"(?:non-numeric|dtype.*object|categorical|columns?).*(?:contain text|not numeric|text.*instead)", re.I),
        "NON_NUMERIC_COLUMNS",
        "Some columns contain text data that the model can't process directly.",
        "Add an Encoding node (Label or One-Hot) to convert text columns into numbers.",
    ),
    (
        re.compile(r"(?:column_configs|No columns configured|encoding.*config)", re.I),
        "NO_ENCODING_CONFIG",
        "No columns have been configured for encoding.",
        "Open the Encoding node settings and select which columns to encode.",
    ),

    # ── Scikit-learn Errors ───────────────────────────────────────────
    (
        re.compile(r"y contains? (?:previously )?unseen labels?:?\s*\[?([^\]]+)\]?", re.I),
        "UNSEEN_LABELS",
        "The test data contains labels ({g1}) that weren't in the training data.",
        "This usually means your dataset is too small or the train/test split was unlucky. Try increasing the training set size or shuffling the data.",
    ),
    (
        re.compile(r"Found input variables with inconsistent numbers of samples:?\s*\[?(\d+)[,\s]+(\d+)", re.I),
        "INCONSISTENT_SAMPLES",
        "The number of data points ({g1}) doesn't match the number of labels ({g2}).",
        "Make sure your features (X) and target (y) have the same number of rows. Check for filtering or missing value steps that might remove rows unevenly.",
    ),
    (
        re.compile(r"Unknown label type:?\s*['\"]?(\w+)", re.I),
        "UNKNOWN_LABEL_TYPE",
        'The target column has an unexpected data type: "{g1}".',
        "For classification, the target should contain categories (text or integers). For regression, it should contain numbers. Check your target column.",
    ),
    (
        re.compile(r"(?:ConvergenceWarning|Maximum iterations? reached|failed to converge)", re.I),
        "CONVERGENCE",
        "The model didn't fully converge (finish learning) within the allowed iterations.",
        "Try increasing 'Max Iterations' in the model settings, or the model may need more features/data to learn the pattern.",
    ),
    (
        re.compile(r"(?:least populated class|too few samples).*?(\d+)\s*(?:member|sample)", re.I),
        "TOO_FEW_SAMPLES",
        "One of the classes has too few samples ({g1}) for the requested split.",
        "Increase your dataset size, reduce the test split ratio, or remove very rare classes.",
    ),
    (
        re.compile(r"(?:n_splits|cross.?validation).*cannot.*(?:greater|more).*(?:number|n_samples)", re.I),
        "TOO_FEW_FOR_CV",
        "There aren't enough samples for the requested cross-validation folds.",
        "Reduce the number of CV folds or add more data to your dataset.",
    ),

    # ── Feature / Column Issues ───────────────────────────────────────
    (
        re.compile(r"(?:feature_names|columns?).*(?:mismatch|don't match|differ)", re.I),
        "FEATURE_MISMATCH",
        "The columns in the test data don't match the training data.",
        "Make sure you're not adding/removing columns between the split and the model. Check your pipeline connections.",
    ),
    (
        re.compile(r"(?:Reshape|Expected \d+D|shape)", re.I),
        "SHAPE_MISMATCH",
        "The data shape doesn't match what the model expects.",
        "Check that your pipeline is correctly connected and that no nodes are filtering columns unexpectedly.",
    ),

    # ── Image Pipeline ────────────────────────────────────────────────
    (
        re.compile(r"(?:no images?|0 images?|images? not found|empty.*image)", re.I),
        "NO_IMAGES",
        "No images were found in the dataset.",
        "Make sure your image dataset has been loaded correctly and the folder path is valid.",
    ),
    (
        re.compile(r"(?:out of memory|OOM|memory.*(?:error|exceeded)|ResourceExhausted)", re.I),
        "OUT_OF_MEMORY",
        "The server ran out of memory processing the data.",
        "Try reducing the image size, batch size, or dataset size in the node settings.",
    ),

    # ── Node Configuration ────────────────────────────────────────────
    (
        re.compile(r"(?:required.*input|no.*input|missing.*input|upstream.*not.*found)", re.I),
        "MISSING_INPUT",
        "This node is missing a required input connection.",
        "Connect the required input node(s) to this node using edges in the pipeline.",
    ),
    (
        re.compile(r"(?:not configured|configuration missing|settings? missing|Some required settings)", re.I),
        "NOT_CONFIGURED",
        "This node hasn't been configured yet.",
        "Double-click the node to open its settings and fill in the required fields.",
    ),

    # ── Pipeline Structure ────────────────────────────────────────────
    (
        re.compile(r"(?:cycle|circular|cyclic).*(?:dependency|detected|graph)", re.I),
        "CYCLE_DETECTED",
        "Your pipeline has a circular connection (a loop).",
        "Remove the edge that creates the loop. Data should flow in one direction only.",
    ),
    (
        re.compile(r"(?:no.*nodes?|empty.*pipeline|pipeline.*empty)", re.I),
        "EMPTY_PIPELINE",
        "The pipeline has no nodes to execute.",
        "Add some nodes to the canvas and connect them before running.",
    ),

    # ── Timeout / Server ──────────────────────────────────────────────
    (
        re.compile(r"(?:timeout|timed? out|took too long)", re.I),
        "TIMEOUT",
        "The operation took too long and was stopped.",
        "Try with a smaller dataset or simpler model settings. Large datasets may need more time.",
    ),
]


def format_error(
    exception: Exception,
    node_type: Optional[str] = None,
    node_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a raw exception into a structured, student-friendly error dict.

    Returns:
        {
            "friendly_message": "Plain English description",
            "suggestion": "What to do about it",
            "technical_detail": "Original error text for debugging",
            "error_code": "UPPER_SNAKE identifier",
            "node_type": "the node type if available",
        }
    """
    raw = str(exception)
    label = node_label or node_type or "this step"

    for pattern, code, message_tpl, suggestion_tpl in ERROR_PATTERNS:
        match = pattern.search(raw)
        if match:
            # Build template context
            ctx = {
                "match": match.group(0),
                "node_label": label,
                "node_type": node_type or "unknown",
            }
            # Add numbered groups
            for i, g in enumerate(match.groups(), 1):
                ctx[f"g{i}"] = g or ""

            try:
                friendly = message_tpl.format(**ctx)
                suggestion = suggestion_tpl.format(**ctx)
            except (KeyError, IndexError):
                friendly = message_tpl
                suggestion = suggestion_tpl

            return {
                "friendly_message": friendly,
                "suggestion": suggestion,
                "technical_detail": raw,
                "error_code": code,
                "node_type": node_type,
            }

    # ── Fallback: no pattern matched ──────────────────────────────────
    # Try to clean up the raw message a bit
    cleaned = raw
    # Remove "Node execution failed [node_type]: " prefix
    prefix_match = re.match(r"Node execution failed \[[\w_]+\]:\s*(.+)", raw)
    if prefix_match:
        cleaned = prefix_match.group(1)

    return {
        "friendly_message": f"Something went wrong while running \"{label}\".",
        "suggestion": "Check your node settings and input connections. If the problem persists, try a different configuration.",
        "technical_detail": cleaned,
        "error_code": "UNKNOWN",
        "node_type": node_type,
    }
