# Visual-ML Project Reference

> Read this FIRST before making any changes. This file tells you where things live, how they connect, and the exact pattern for every kind of change.

---

## Project Layout

```
Visual-ML/
├── client/          React + TypeScript + Vite frontend
│   └── src/
│       ├── app/                  App shell / layout
│       ├── components/
│       │   ├── playground/       ← Main ML canvas UI (all node UI lives here)
│       │   ├── common/           Shared components
│       │   └── projects/         Project management
│       ├── config/
│       │   ├── nodeDefinitions.ts  ← MASTER node registry (frontend)
│       │   ├── mlAlgorithms.ts     ML algorithm node defs
│       │   └── genaiNodes.ts       GenAI node defs
│       ├── features/             Feature modules (app-builder, gamification, mentor)
│       ├── hooks/                React Query mutations & queries
│       ├── lib/                  axios client, API helpers, export utils
│       ├── pages/                Route-level pages
│       ├── store/
│       │   ├── playgroundStore.ts  ← ALL pipeline state (nodes, edges, results)
│       │   ├── authStore.ts
│       │   └── uiStore.ts
│       └── types/
│           └── pipeline.ts         ← NodeType union + shared types
│
├── server/          Python FastAPI backend
│   └── app/
│       ├── ml/
│       │   ├── engine.py           ← NODE REGISTRY (backend) + pipeline runner
│       │   └── nodes/              ← One file per node type
│       │       ├── base.py         BaseNode, NodeInput, NodeOutput
│       │       ├── genai/          GenAI nodes
│       │       └── results_and_metrics/
│       ├── api/v1/
│       │   ├── pipelines.py        Main execution endpoints
│       │   ├── datasets.py
│       │   ├── projects.py
│       │   └── ...                 (8 more route files)
│       ├── core/                   Config, logging, security
│       ├── db/                     SQLAlchemy session
│       ├── models/                 ORM models
│       └── schemas/                Pydantic schemas
│
└── .claude/
    └── PROJECT_SUMMARY.md  ← THIS FILE
```

---

## The 5-File Pattern: How to Add / Modify Any Node

Every node type touches exactly these files. Do them in order.

### 1. `client/src/types/pipeline.ts`
Add the new node type string to the `NodeType` union.
```typescript
| "my_new_node"
```

### 2. `client/src/config/nodeDefinitions.ts`
Add the node definition object inside the correct category array.
```typescript
{
  type: "my_new_node",
  label: "Human Label",
  description: "Short description",
  category: "category-id",   // see Category IDs below
  icon: SomeIcon,            // import from lucide-react
  color: "#HEX",
  defaultConfig: { field1: "default" },
  configFields: [
    { name: "field1", label: "Label", type: "select", options: [...], defaultValue: "x" },
    { name: "field2", label: "Label", type: "text", required: true, autoFill: true },
    { name: "flag",   label: "Label", type: "checkbox", defaultValue: false },
    { name: "num",    label: "Label", type: "number", min: 0, max: 100, step: 1 },
  ],
}
```
**ConfigField types:** `text | select | checkbox | number`
**autoFill: true** — value is auto-populated by the edge-connection system from upstream node output.

### 3. `client/src/components/playground/Canvas.tsx`
Add the node type to the `nodeTypes` const (~line 50):
```typescript
my_new_node: MLNode,
```

### 4. `client/src/components/playground/MLNode.tsx`
Add to the `viewNodeTypes` array (~line 54) if the node has results to view:
```typescript
"my_new_node",
```

### 5a. `client/src/components/playground/ConfigModal.tsx`
Only needed if the node needs a **custom config UI** (camera, dataset toggle, etc.).
Add a ternary branch before the generic `nodeDef.configFields` fallback (~line 1662):
```typescript
) : node.type === "my_new_node" ? (
  <MyNewNodeConfigBlock config={config} onFieldChange={handleFieldChange} />
) : nodeDef.configFields && nodeDef.configFields.length > 0 ? (
```
Define `function MyNewNodeConfigBlock(...)` above the `ConfigModal` export.

### 5b. `client/src/components/playground/ViewNodeModal.tsx`
Add a result explorer block (~line 273):
```typescript
{nodeType === "my_new_node" && (
  <MyNewNodeExplorer result={nodeResult} renderResults={() => renderTableView(nodeResult)} />
)}
```
Then create `MyNewNodeExplorer.tsx` in the playground folder.

### 6. `server/app/ml/nodes/my_new_node.py`
```python
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from pydantic import Field
from typing import Optional

class MyNewNodeInput(NodeInput):
    field1: str = Field("default", description="...")

class MyNewNodeOutput(NodeOutput):
    result_field: str = Field(...)

class MyNewNode(BaseNode):
    node_type = "my_new_node"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.DATA_SOURCE,  # or PREPROCESSING, MODEL, etc.
            primary_output_field="result_field",
            output_fields={"result_field": "description"},
            requires_input=False,  # True if it needs upstream data
        )

    def get_input_schema(self): return MyNewNodeInput
    def get_output_schema(self): return MyNewNodeOutput

    async def _execute(self, input_data: MyNewNodeInput) -> MyNewNodeOutput:
        # ... logic ...
        return MyNewNodeOutput(node_type=self.node_type, execution_time_ms=0, ...)
```

### 7. `server/app/ml/engine.py`
Inside `_auto_discover_nodes()`, add the import and registry entry:
```python
from app.ml.nodes.my_new_node import MyNewNode
# ...
nodes_to_register = {
    # ... existing entries ...
    "my_new_node": MyNewNode,
}
```

---

## Current Node Types (Full List)

### Data Sources
| Type | Label | Frontend File | Backend File |
|------|-------|---------------|--------------|
| `upload_file` | Upload File | nodeDefinitions.ts | nodes/upload.py |
| `select_dataset` | Select Dataset | nodeDefinitions.ts | nodes/select.py |
| `sample_dataset` | Sample Dataset | nodeDefinitions.ts | nodes/sample_dataset.py |

### View Nodes
| Type | Label |
|------|-------|
| `table_view` | Table View |
| `data_preview` | Data Preview |
| `statistics_view` | Statistics |
| `column_info` | Column Info |
| `chart_view` | Chart |

### Preprocessing
| Type | Label | Backend File |
|------|-------|--------------|
| `missing_value_handler` | Missing Values | missing_value_handler.py |
| `encoding` | Encoding | encoding.py |
| `transformation` | Transformation | transformation.py |
| `scaling` | Scaling | scaling.py |
| `feature_selection` | Feature Selection | feature_selection.py |
| `split` | Split | split.py |

### ML Algorithms
| Type | Backend File |
|------|--------------|
| `linear_regression` | linear_regression_node.py |
| `logistic_regression` | logistic_regression_node.py |
| `decision_tree` | decision_tree_node.py |
| `random_forest` | random_forest_node.py |
| `mlp_classifier` | mlp_classifier_node.py |
| `mlp_regressor` | mlp_regressor_node.py |

### Results & Metrics
`r2_score` · `mse_score` · `rmse_score` · `mae_score` · `confusion_matrix`

### Image Pipeline (in execution order)
```
image_dataset  →  image_preprocessing  →  image_split  →  cnn_classifier  →  image_predictions
```
| Type | Backend File | Special Notes |
|------|--------------|---------------|
| `image_dataset` | image_dataset_node.py | Merged: source=builtin or source=camera. Camera UI via CameraCapturePanel inside ImageDatasetConfigBlock in ConfigModal |
| `image_preprocessing` | image_preprocessing_node.py | normalize_method: none/minmax/standard/divide_255 |
| `image_split` | image_split_node.py | test_size, stratify, random_seed |
| `cnn_classifier` | cnn_classifier_node.py | Trains model, outputs model_path |
| `image_predictions` | image_predictions_node.py | Uses test_dataset_id + model_path from split+cnn |

### GenAI
`llm_node` · `system_prompt` · `chatbot_node` · `example_node`
Defined in `client/src/config/genaiNodes.ts`, handled by `GenAIConfigPanel.tsx`.

### Activity Nodes (standalone learning)
`activity_gradient_descent` · `activity_bias_variance` · `activity_knn_playground`
`activity_decision_boundary` · `activity_neural_network`

---

## Category IDs (use in nodeDefinitions.ts)
```
"data-source"          Data Sources
"view"                 View nodes
"preprocess"           Preprocess Data
"feature-engineering"  Feature Engineering
"target-split"         Target & Split
"ml-algorithms"        ML Algorithms
"image-pipeline"       Image Pipeline
"result"               Results & Metrics
"genai"                GenAI
"activity"             Interactive Activities
```

---

## ConfigModal Ternary Chain (order matters)

ConfigModal.tsx dispatches node config UI in this order (simplified):
```
1. upload_file          → dataset upload panel
2. select_dataset       → dataset picker + tabbed config
3. sample_dataset       → sample dataset picker
4. split                → SplitConfigPanel
5. preprocess/FE nodes  → FeatureEngineeringConfigPanel
6. ML algorithms        → MLAlgorithmConfigPanel
7. result/metrics nodes → ResultMetricsConfigPanel
8. genai nodes          → GenAIConfigPanel
9. image_dataset        → ImageDatasetConfigBlock  ← custom (builtin+camera toggle)
10. (fallback)          → generic nodeDef.configFields renderer
```
`image_predictions` uses the generic fallback (has autoFill text fields only).

---

## ViewNodeModal Explorer Map

| Node Type | Explorer Component | File |
|-----------|-------------------|------|
| `image_dataset` | ImageDatasetExplorer | ImageDatasetExplorer.tsx |
| `image_preprocessing` | ImagePreprocessingExplorer | ImagePreprocessingExplorer.tsx |
| `image_split` | ImageSplitExplorer | ImageSplitExplorer.tsx |
| `cnn_classifier` | CNNClassifierExplorer | CNNClassifierExplorer.tsx |
| `image_predictions` | ImagePredictionsExplorer | ImagePredictionsExplorer.tsx |
| `split` | SplitExplorer | SplitExplorer.tsx |
| `missing_value_handler` | MissingValueExplorer | MissingValueExplorer.tsx |
| `encoding` | EncodingExplorer | EncodingExplorer.tsx |
| `scaling` | ScalingExplorer | ScalingExplorer.tsx |
| `feature_selection` | FeatureSelectionExplorer | FeatureSelectionExplorer.tsx |
| `linear_regression` | LinearRegressionExplorer | LinearRegressionExplorer.tsx |
| `logistic_regression` | LogisticRegressionExplorer | LogisticRegressionExplorer.tsx |

**renderTableView** is the generic fallback shown in each explorer's "Results" tab.
It reads `result.data[]` and `result.columns[]`. Image nodes have 0 rows here by design (data is saved to CSV, not returned inline).

---

## Key State: playgroundStore.ts

```typescript
// Reading node data
const { getNodeById, nodes, edges } = usePlaygroundStore();

// Updating node config (saves to store AND upstream)
updateNodeConfig(nodeId, { field: value });

// Execution results
nodeExecutionStatus[nodeId]  // "pending" | "running" | "success" | "error"
executionResults[nodeId]     // the raw result object from backend

// Camera dataset build (in ConfigModal)
buildCameraDataset(payload)  // POSTs to /ml/camera/dataset, stores dataset_id in node config
```

---

## Backend Node Base Pattern

```python
class NodeCategory:
    DATA_SOURCE = "data_source"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL = "model"
    EVALUATION = "evaluation"

class NodeInput(BaseModel):
    # All config fields that come from the frontend node config

class NodeOutput(BaseModel):
    node_type: str         # always set to self.node_type
    execution_time_ms: int # always 0 is fine, engine sets it
    # ... your output fields

class MyNode(BaseNode):
    node_type = "my_node"   # must match frontend type string exactly

    async def _execute(self, input_data: MyNodeInput) -> MyNodeOutput:
        ...
```

**Output field auto-wiring:** Fields listed in `metadata.output_fields` are auto-propagated to downstream nodes that have matching `autoFill: true` configFields. The field name must match between the output schema and the downstream node's configField `name`.

---

## API Endpoints (relevant ones)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/pipelines/execute` | Execute full pipeline |
| POST | `/api/v1/pipelines/execute/stream` | Execute with SSE streaming |
| POST | `/api/v1/ml/camera/dataset` | Build camera dataset from pixel arrays |
| POST | `/api/v1/ml/camera/predict` | Live camera prediction |
| GET  | `/api/v1/datasets/` | List user datasets |
| POST | `/api/v1/datasets/upload` | Upload dataset file |

---

## Common Gotchas

1. **Adding a node type?** Must add to ALL 7 places (types, nodeDefinitions, Canvas, MLNode, ConfigModal or fallback, ViewNodeModal, backend engine). Missing any one = runtime error.

2. **Removing a node type?** Same 7 places. Also check for imports in ViewNodeModal and ConfigModal — they crash at render if an import points to a deleted file.

3. **`autoFill` fields** are populated automatically when a pipeline edge is drawn. The upstream node's output field name must exactly match the downstream node's configField `name`.

4. **ConfigModal ternary ordering** — more specific checks (single node.type) must come before array `.includes()` checks that might overlap.

5. **`ImageDatasetConfigBlock`** is defined inline in ConfigModal.tsx (not a separate file) — it handles the source toggle between builtin datasets and camera capture.

6. **Camera capture flow:** User captures images in `CameraCapturePanel` → clicks "Build Dataset" → `buildCameraDataset()` POSTs to `/ml/camera/dataset` → gets `dataset_id` back → stored in `image_dataset` node config as `source=camera, dataset_id=...` → on pipeline run, `image_dataset_node.py` detects `source=camera` and reads from the saved CSV.

7. **Image pipeline "0 rows" in Results tab** — intentional. The CSV has millions of cells; only metadata + sample images are returned. The Gallery/Distribution/Quiz tabs have the real data.

8. **`split_ratios`** in ImageSplitExplorer expects `train_count` and `test_count` — added to `_generate_split_ratios()` in `image_split_node.py`.

---

## Environment

- **Frontend dev:** `cd client && npm run dev` → http://localhost:5173
- **Backend dev:** `cd server && uvicorn app.main:app --reload` → http://localhost:8000
- **API docs:** http://localhost:8000/docs
- **Client env:** `client/.env` — key var: `VITE_API_URL`
- **Server env:** `server/.env` — DB connection, upload dir, secret key, etc.
