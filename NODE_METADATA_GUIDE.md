# Quick Reference: Adding Metadata to Nodes

## Step-by-Step Guide

### 1. Import Required Types

Add to imports at top of file:

```python
from app.ml.nodes.base import (
    BaseNode,
    NodeInput,
    NodeOutput,
    NodeMetadata,      # ← Add this
    NodeCategory       # ← Add this
)
```

### 2. Add @property metadata

Add this after `node_type` and before `get_input_schema()`:

```python
@property
def metadata(self) -> NodeMetadata:
    """Return node metadata for DAG execution."""
    return NodeMetadata(
        category=NodeCategory.___,  # Choose from categories below
        primary_output_field="___",  # Main output field name
        output_fields={
            "field1": "Description",
            "field2": "Description"
        },
        requires_input=True/False,
        can_branch=True/False,
        produces_dataset=True/False,
        max_inputs=None,  # or specific number
        allowed_source_categories=[  # Optional
            NodeCategory.DATA_SOURCE,
            NodeCategory.PREPROCESSING
        ]
    )
```

## NodeCategory Options

```python
NodeCategory.DATA_SOURCE          # upload_file, select_dataset
NodeCategory.PREPROCESSING        # encoding, scaling, missing_value_handler
NodeCategory.FEATURE_ENGINEERING  # feature_selection, transformation
NodeCategory.DATA_TRANSFORM       # split (creates train/test)
NodeCategory.ML_ALGORITHM         # linear_regression, decision_tree
NodeCategory.METRIC               # r2_score, mae_score
NodeCategory.VIEW                 # table_view, chart_view
NodeCategory.UTILITY              # other utilities
```

## Examples by Node Type

### Data Source Nodes (upload, select)

```python
@property
def metadata(self) -> NodeMetadata:
    return NodeMetadata(
        category=NodeCategory.DATA_SOURCE,
        primary_output_field="dataset_id",
        requires_input=False,  # Source nodes have no inputs
        can_branch=True,
        produces_dataset=True,
        max_inputs=0  # Cannot accept inputs
    )
```

### Preprocessing Nodes (encoding, scaling, missing_value_handler, transformation, feature_selection)

```python
@property
def metadata(self) -> NodeMetadata:
    return NodeMetadata(
        category=NodeCategory.PREPROCESSING,
        primary_output_field="encoded_dataset_id",  # Change per node
        # OR: "scaled_dataset_id", "transformed_dataset_id", etc.
        requires_input=True,
        can_branch=True,
        produces_dataset=True,
        allowed_source_categories=[
            NodeCategory.DATA_SOURCE,
            NodeCategory.PREPROCESSING  # Can chain preprocessing
        ]
    )
```

**Field Names by Node:**

- MissingValueHandlerNode: `preprocessed_dataset_id`
- EncodingNode: `encoded_dataset_id`
- TransformationNode: `transformed_dataset_id`
- ScalingNode: `scaled_dataset_id`
- FeatureSelectionNode: `selected_dataset_id`

### Data Transform Nodes (split)

```python
@property
def metadata(self) -> NodeMetadata:
    return NodeMetadata(
        category=NodeCategory.DATA_TRANSFORM,
        primary_output_field="train_dataset_id",  # Primary output
        output_fields={
            "train_dataset_id": "Training dataset",
            "test_dataset_id": "Test dataset"  # Secondary output
        },
        requires_input=True,
        can_branch=True,  # Both outputs can feed different nodes
        produces_dataset=True,
        allowed_source_categories=[
            NodeCategory.DATA_SOURCE,
            NodeCategory.PREPROCESSING
        ]
    )
```

### ML Algorithm Nodes (linear_regression, logistic_regression, decision_tree, random_forest)

```python
@property
def metadata(self) -> NodeMetadata:
    return NodeMetadata(
        category=NodeCategory.ML_ALGORITHM,
        primary_output_field="model_id",
        output_fields={
            "model_id": "Trained model identifier",
            "model_path": "Path to saved model",
            "training_metrics": "Performance metrics"
        },
        requires_input=True,
        can_branch=True,  # Can feed multiple metric nodes
        produces_dataset=False,  # Produces model, not dataset
        max_inputs=1,
        allowed_source_categories=[
            NodeCategory.DATA_TRANSFORM,  # From split
            NodeCategory.PREPROCESSING
        ]
    )
```

### Metric Nodes (r2_score, mae_score, mse_score, rmse_score)

```python
@property
def metadata(self) -> NodeMetadata:
    return NodeMetadata(
        category=NodeCategory.METRIC,
        primary_output_field="score",  # Or "r2_score", "mae_score", etc.
        output_fields={
            "score": "Calculated metric value",
            "metric_name": "Name of metric"
        },
        requires_input=True,
        can_branch=True,  # Can feed into views
        produces_dataset=False,
        max_inputs=1,
        allowed_source_categories=[
            NodeCategory.ML_ALGORITHM  # Must come from trained model
        ]
    )
```

### View Nodes (table_view, data_preview, statistics_view, column_info, chart_view)

```python
@property
def metadata(self) -> NodeMetadata:
    return NodeMetadata(
        category=NodeCategory.VIEW,
        primary_output_field=None,  # Views don't produce datasets
        output_fields={
            "data": "View data",
            "view_type": "Type of visualization"
        },
        requires_input=True,
        can_branch=False,  # Terminal nodes
        produces_dataset=False,
        max_inputs=1,
        allowed_source_categories=[
            NodeCategory.DATA_SOURCE,
            NodeCategory.PREPROCESSING,
            NodeCategory.DATA_TRANSFORM
        ]
    )
```

## Checklist for Each Node

- [ ] Import `NodeMetadata` and `NodeCategory`
- [ ] Add `@property metadata` after `node_type`
- [ ] Choose correct `NodeCategory`
- [ ] Set `primary_output_field` (check output schema)
- [ ] Set `requires_input` (True for most, False for source nodes)
- [ ] Set `can_branch` (False only for terminal view nodes)
- [ ] Set `produces_dataset` (False for models/metrics/views)
- [ ] Set `max_inputs` (0 for sources, 1 for most, None for multi-input)
- [ ] Test node still executes correctly

## Common Mistakes to Avoid

❌ **Wrong**: Forgetting to import `NodeMetadata` and `NodeCategory`
✅ **Right**: Import both at top of file

❌ **Wrong**: Setting `max_inputs=0` for preprocessing nodes
✅ **Right**: `max_inputs=None` or `max_inputs=1`

❌ **Wrong**: Setting `requires_input=False` for preprocessing
✅ **Right**: `requires_input=True` (only source nodes are False)

❌ **Wrong**: Setting `can_branch=False` for preprocessing
✅ **Right**: `can_branch=True` (only view nodes are False)

❌ **Wrong**: Using wrong field name in `primary_output_field`
✅ **Right**: Check the output schema - use exact field name

## Testing Your Metadata

After adding metadata, verify:

```python
# 1. Node can be instantiated
node = YourNode()

# 2. Metadata returns NodeMetadata
assert isinstance(node.metadata, NodeMetadata)

# 3. Category is valid
assert node.metadata.category in NodeCategory

# 4. Primary output field exists in output schema
output_schema = node.get_output_schema()
if node.metadata.primary_output_field:
    assert node.metadata.primary_output_field in output_schema.__fields__
```

## Need Help?

Check these implemented examples:

- Data Source: [select.py](server/app/ml/nodes/select.py) - SelectDatasetNode
- Preprocessing: [encoding.py](server/app/ml/nodes/encoding.py) - EncodingNode
- ML Algorithm: [linear_regression_node.py](server/app/ml/nodes/linear_regression_node.py)
- View: [view.py](server/app/ml/nodes/view.py) - TableViewNode (line 245)
