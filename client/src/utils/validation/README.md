# Pipeline Validation System

## Overview

The validation system ensures pipeline integrity before execution by checking node connections, configurations, and preventing invalid workflows.

## Architecture

```
utils/validation/
├── pipelineValidation.ts  # Core validation logic
├── index.ts               # Export barrel
└── README.md             # This file
```

## Node Categories

### DATA_SOURCE

- `upload_file` - Upload CSV files
- `select_dataset` - Select from saved datasets
- `load_url` - Load data from URL

### VIEW

- `table_view` - Display data in table format
- `data_preview` - Preview dataset
- `statistics_view` - Show statistical summary
- `column_info` - Display column information
- `chart_view` - Render charts (Bar, Line, Scatter, Histogram, Pie)

### PROCESSING

- `preprocess` - Data preprocessing
- `split` - Train/test split
- `train` - Model training
- `evaluate` - Model evaluation

### FEATURE_ENGINEERING

- `feature_engineering` - Feature transformation

## Validation Rules

### 1. View Node Connections (`validateViewNodeConnections`)

**Rule**: View nodes MUST connect to Data Source nodes only

**Valid**:

```
[Upload File] ──→ [Table View]
[Select Dataset] ──→ [Chart View]
```

**Invalid**:

```
[Table View] ──→ [Chart View]  ❌
[Preprocess] ──→ [Table View]  ❌
```

**Error Type**: `error` (blocks execution)

### 2. Node Configuration (`validateNodeConfiguration`)

**Rule**: All nodes must have required configuration

**Checks**:

- Nodes have non-empty config object
- Required fields are present

**Error Type**: `warning` (allows execution with confirmation)

### 3. Circular Dependencies (`validateNoCircularDependencies`)

**Rule**: No cycles in the pipeline graph

**Example**:

```
[Node A] ──→ [Node B] ──→ [Node C]
    ↑                        ↓
    └────────────────────────┘  ❌
```

**Error Type**: `error` (blocks execution)

## Usage

### In Components

```typescript
import { validatePipeline } from "../../utils/validation";

const validationResult = validatePipeline(nodes, edges);

if (!validationResult.isValid) {
  setValidationErrors(validationResult.errors);
  return;
}
```

### Validation Result

```typescript
interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

interface ValidationError {
  type: "error" | "warning";
  nodeId: string;
  message: string;
  suggestion?: string;
}
```

## UI Integration

### ValidationDialog Component

Located: `client/src/components/playground/ValidationDialog.tsx`

**Features**:

- Separates errors and warnings
- Shows helpful suggestions
- "Proceed Anyway" button for warnings
- Blocks execution for errors

**Error Display**:

- 🔴 Red alerts for errors (blocking)
- 🟡 Yellow alerts for warnings (non-blocking)

## Execution Flow

```
User clicks "Execute Pipeline"
         ↓
validatePipeline(nodes, edges)
         ↓
   Has errors? ──Yes──→ Show ValidationDialog (No "Proceed")
         ↓                      ↓
        No               User clicks "Close"
         ↓                      ↓
  Has warnings? ──Yes──→ Show ValidationDialog (With "Proceed")
         ↓                      ↓
        No               User chooses: Close or Proceed
         ↓                      ↓
    Execute                 Execute
    Pipeline               Pipeline
```

## Adding New Validations

1. **Create validation function** in `pipelineValidation.ts`:

```typescript
function validateYourRule(nodes: MLNode[], edges: Edge[]): ValidationError[] {
  const errors: ValidationError[] = [];

  // Your validation logic

  return errors;
}
```

2. **Add to `validatePipeline`**:

```typescript
export function validatePipeline(
  nodes: MLNode[],
  edges: Edge[],
): ValidationResult {
  const errors: ValidationError[] = [
    ...validateViewNodeConnections(nodes, edges),
    ...validateNodeConfiguration(nodes),
    ...validateNoCircularDependencies(nodes, edges),
    ...validateYourRule(nodes, edges), // Add here
  ];

  return {
    isValid: !errors.some((e) => e.type === "error"),
    errors,
  };
}
```

3. **Export in `index.ts`** (optional):

```typescript
export {
  validatePipeline,
  validateYourRule, // Add here
  // ...
};
```

## Best Practices

1. **Error vs Warning**:
   - Use `"error"` for validation failures that MUST block execution
   - Use `"warning"` for issues the user should know about but can proceed

2. **Helpful Messages**:
   - Be specific about what's wrong
   - Provide actionable suggestions
   - Reference node names when possible

3. **Performance**:
   - Validations run on every execute click
   - Keep validation logic efficient
   - Avoid expensive operations

## Testing Validation

### Test Case 1: View-to-View Connection

1. Drag `Table View` and `Chart View` nodes
2. Connect `Table View` → `Chart View`
3. Click "Execute Pipeline"
4. **Expected**: Error dialog blocks execution

### Test Case 2: Valid Data Flow

1. Drag `Upload File`, `Table View`, `Chart View`
2. Connect `Upload File` → `Table View`
3. Connect `Upload File` → `Chart View`
4. Click "Execute Pipeline"
5. **Expected**: No validation errors (if nodes configured)

### Test Case 3: Unconfigured Node

1. Drag `Upload File` node
2. Don't configure it
3. Drag `Table View` and connect
4. Click "Execute Pipeline"
5. **Expected**: Warning dialog with "Proceed Anyway" option

## Future Enhancements

- [ ] Validate required inputs for each node type
- [ ] Check data type compatibility between connections
- [ ] Validate ML algorithm parameters
- [ ] Detect disconnected nodes
- [ ] Warn about unused data sources
- [ ] Validate feature engineering transformations
- [ ] Check for duplicate node names
- [ ] Validate dataset compatibility

## Troubleshooting

### Validation not running?

- Check if `validatePipeline` is called in `handleExecute`
- Verify imports are correct

### Dialog not showing?

- Check `validationErrors` state is set
- Verify `ValidationDialog` is rendered in return statement
- Check browser console for React errors

### False positives?

- Review node category definitions in `NODE_CATEGORIES`
- Check edge direction (source → target)
- Verify node types match category strings exactly
