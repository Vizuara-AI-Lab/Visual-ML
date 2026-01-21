# ML Pipeline Playground

A visual ML pipeline builder using React Flow that allows users to build, configure, and execute machine learning pipelines.

## Features

### Available Nodes

1. **Upload Data** 📤
   - Upload CSV datasets
   - Validate file format and size
   - Stores dataset for pipeline processing

2. **Preprocess Data** 🧹
   - Handle missing values (drop, mean, median, mode, fill)
   - Text preprocessing with TF-IDF
   - Numeric feature scaling with StandardScaler
   - Configurable preprocessing strategies

3. **Split** ✂️
   - Train/Validation/Test split
   - Configurable ratios (must sum to 1.0)
   - Stratified splitting support
   - Reproducible with random seed

4. **Train** 🎓
   - **Linear Regression** (for regression tasks)
     - Hyperparameters: fit_intercept, copy_X, n_jobs
   - **Logistic Regression** (for classification tasks)
     - Hyperparameters: C, penalty, solver, max_iter, tol, class_weight
   - Model versioning and persistence
   - Training metrics tracking

5. **View Result** 📊
   - Evaluate model performance
   - **Regression metrics**: MAE, MSE, RMSE, R²
   - **Classification metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
   - Per-class metrics for multi-class classification

## How to Use

### Building a Pipeline

1. **Drag nodes** from the left palette onto the canvas
2. **Connect nodes** by dragging from the output handle (bottom) of one node to the input handle (top) of another
3. **Configure nodes** by clicking on them to open the configuration panel
4. **Save configurations** for each node
5. **Validate** the pipeline to check for errors
6. **Run** the pipeline to execute

### Example Pipeline Flow

```
Upload Data → Preprocess Data → Split → Train → View Result
```

### Node Configuration

Each node requires specific configuration:

#### Upload Data

- Select a CSV file from your computer
- Filename is automatically captured

#### Preprocess Data

- Dataset path (from previous Upload node output)
- Target column name
- Missing value strategy
- Text/numeric columns for processing
- Feature scaling options

#### Split

- Dataset path
- Target column
- Train/Val/Test ratios (must sum to 1.0)
- Random seed for reproducibility
- Shuffle and stratify options

#### Train

- Training dataset path (from Split node output)
- Target column
- Task type (regression/classification)
- Algorithm selection
- Hyperparameters configuration
- Optional model name

#### View Result

- Model path (from Train node output)
- Test dataset path (from Split node output)
- Target column
- Task type

## Pipeline Execution

### Validation

- Click **Validate** to check pipeline configuration without execution
- Validates all node configurations
- Checks for missing required fields
- Verifies data flow consistency

### Execution

- Click **Run Pipeline** to execute the complete pipeline
- Nodes are executed in topological order
- Results from each node are passed to connected nodes
- Execution time is tracked
- Results are displayed in the results panel

## API Integration

The playground communicates with the backend via REST API:

- `POST /ml/nodes/run` - Execute a single node
- `POST /ml/pipeline/run` - Execute complete pipeline

All API calls include authentication and error handling.

## Technical Details

### Technologies

- **React Flow** - Visual node editor
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **React Query** - Data fetching (if needed)

### File Structure

```
client/src/
├── types/
│   └── pipeline.ts              # Type definitions
├── lib/
│   └── nodeDefinitions.ts       # Node metadata
├── features/
│   └── playground/
│       ├── api.ts               # API integration
│       └── index.ts             # Exports
├── components/
│   └── playground/
│       ├── MLNode.tsx           # Custom node component
│       ├── NodePalette.tsx      # Draggable node palette
│       └── NodeConfigPanel.tsx  # Configuration UI
└── pages/
    └── playground/
        └── PlayGround.tsx       # Main playground page
```

## Future Enhancements

- [ ] Save/load pipeline configurations
- [ ] Export pipeline results
- [ ] More ML algorithms (SVM, Random Forest, etc.)
- [ ] Real-time dataset preview
- [ ] Pipeline templates
- [ ] Collaborative editing
- [ ] Version control for pipelines
- [ ] Automated hyperparameter tuning
