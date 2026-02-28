# Visual-ML — Knowledge Transfer Document

> **Purpose**: Complete system overview for onboarding new team members. Covers what the platform does, how users interact with it, every page/URL, all features, and how things connect.

---

## 1. What is Visual-ML?

Visual-ML is an **educational interactive platform** that teaches Machine Learning visually. Students build ML pipelines by dragging and connecting nodes on a canvas — no coding required. The platform covers the entire ML workflow: loading data, cleaning it, engineering features, training models, and evaluating results.

**Target audience**: Students learning ML for the first time, guided by an AI mentor with voice support.

**Core idea**: Instead of writing Python code, students visually connect building blocks (nodes) on a canvas to create complete ML workflows. Each node has an interactive explorer with quizzes, charts, and animations to reinforce learning.

---

## 2. Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| Frontend | React 19 + TypeScript + Vite |
| State Management | Zustand (global store) + TanStack React Query (API cache) |
| Canvas/Pipeline | React Flow (@xyflow/react) — drag-drop node graph |
| Styling | Tailwind CSS 4 |
| Backend | Python FastAPI |
| Database | SQLAlchemy ORM (PostgreSQL) |
| ML Engine | Scikit-learn + TensorFlow/Keras |
| File Storage | AWS S3 (datasets) + local uploads folder (processed files) |
| Async Tasks | Celery + Redis |
| AI Mentor | Google Gemini / OpenAI / Anthropic APIs |
| Voice (TTS) | Inworld AI — voice synthesis + voice cloning |
| Auth | JWT tokens in HTTP-only cookies + Google OAuth |

---

## 3. All Pages & URLs

### 3.1 Public Pages (No Login Required)

| URL | Page | What the User Sees |
|-----|------|--------------------|
| `/` | **Landing Page** | Marketing page with hero section, feature showcase, GenAI section, testimonials, and Sign In / Sign Up buttons |
| `/signin` | **Sign In** | Email + password form, Google OAuth button, "Forgot password" link |
| `/signup` | **Sign Up** | Registration form with name, email, password. Has password strength indicator. Redirects to email verification on success |
| `/verify-email` | **Email Verification** | 6-digit OTP input. Auto-focuses between digits. Has resend button with countdown timer |
| `/forgot-password` | **Forgot Password** | Email input to request password reset. Shows success state with auto-redirect countdown |
| `/shared/:shareToken` | **Shared Project** | Read-only view of someone's ML pipeline. Shows the canvas with nodes/edges. May allow cloning depending on permissions |
| `/app/:slug` | **Published App** | Public-facing custom app built from an ML pipeline. No login needed to use it |
| `/admin/login` | **Admin Login** | Separate admin login form |

### 3.2 Protected Pages (Login Required)

| URL | Page | What the User Sees |
|-----|------|--------------------|
| `/dashboard` | **Student Dashboard** | Welcome banner, stat cards (projects count, datasets, shared), project grid with search/filter, "New Project" button |
| `/playground/:projectId` | **Pipeline Builder** | The main workspace — canvas with sidebar, toolbar, config modals. This is where 90% of the user's time is spent |
| `/profile` | **User Profile** | Edit personal info, upload profile picture, change password, voice preferences, gamification dashboard (XP bar, level, badges) |
| `/app-builder/:appId` | **App Builder** | Drag-and-drop UI builder to create custom web apps from ML pipelines |
| `/admin/dashboard` | **Admin Dashboard** | Student management — search, filter, toggle premium/active status |
| `/admin/students/:id` | **Student Detail** | Individual student's profile, activity, and statistics |

### 3.3 How Authentication Works

1. **Sign Up flow**: `/signup` → email sent → `/verify-email` (enter OTP) → `/dashboard`
2. **Sign In flow**: `/signin` → `/dashboard`
3. **Protection**: All `/dashboard`, `/playground`, `/profile` routes check for `user` in localStorage. If missing → redirect to `/signin`
4. **Admin protection**: Admin routes additionally check `userRole === "ADMIN"`
5. **Session**: JWT access + refresh tokens stored in HTTP-only cookies

---

## 4. The Pipeline Builder (Playground) — The Core Feature

**URL**: `/playground/:projectId`

This is the heart of the application. Here's what the user sees:

### 4.1 Layout

```
┌─────────────────────────────────────────────────────┐
│  Toolbar (project name, run button, share, export)  │
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│ Sidebar  │           Canvas                         │
│ (nodes   │        (React Flow)                      │
│  to drag)│                                          │
│          │    [Node] ──→ [Node] ──→ [Node]          │
│          │                                          │
│          │                                          │
├──────────┴──────────────────────────────────────────┤
│              Results Drawer (bottom)                 │
└─────────────────────────────────────────────────────┘
                              + Floating AI Mentor (bottom-right)
```

### 4.2 Sidebar — Node Palette

The sidebar organizes all available nodes into categories. Users **drag nodes** from the sidebar onto the canvas.

**Categories and their nodes:**

| Category | Nodes |
|----------|-------|
| **Templates** | Pre-built pipelines (see 4.3) |
| **Data Sources** | Upload Dataset, Select Dataset, Sample Dataset |
| **View** | Table View, Data Preview, Statistics View, Column Info, Chart View |
| **Preprocess Data** | Missing Value Handler |
| **Feature Engineering** | Encoding, Scaling/Normalization, Feature Selection |
| **Target & Split** | Target & Split |
| **ML Algorithms** | Linear Regression, Logistic Regression, Decision Tree, Random Forest, MLP Classifier, MLP Regressor, K-Means Clustering |
| **Image Pipeline** | Image Dataset, Image Split, Image Predictions |
| **Results & Metrics** | R² Score, MSE, RMSE, MAE, Confusion Matrix |
| **GenAI** | LLM Provider, System Prompt, Examples (Few-Shot), Chatbot |
| **Activities** | 11 interactive learning modules (see Section 7) |

### 4.3 Templates — One-Click Pipeline Setup

Templates instantly create a complete pipeline on the canvas. The user can then customize each node's config.

| Template | Pipeline Created |
|----------|-----------------|
| **Linear Regression** | Upload → Table View → Missing Values → Encoding → Split → Linear Regression → R², MSE, RMSE, MAE |
| **Logistic Regression** | Upload → Table View → Missing Values → Encoding → Split → Logistic Regression → Confusion Matrix |
| **Decision Tree** | Upload → Table View → Missing Values → Encoding → Split → Decision Tree → Confusion Matrix, RMSE |
| **Random Forest** | Upload → Table View → Missing Values → Encoding → Scaling → Split → Random Forest → Confusion Matrix, MAE |
| **GenAI Chatbot** | LLM Provider + System Prompt + Examples → Chatbot |

### 4.4 How Users Build a Pipeline

1. **Drag a node** from sidebar onto canvas (or click a template)
2. **Click a node** to open its **Config Modal** — set parameters (dataset, columns, hyperparameters, etc.)
3. **Connect nodes** by dragging from one node's output handle to another's input handle
4. **Click "Run Pipeline"** on the toolbar — the backend executes nodes in topological order
5. **View Results** — click any executed node to open the **View Modal** with interactive explorer (charts, quizzes, metrics)

### 4.5 Auto-Fill System

Many config fields have `autoFill: true` — they automatically receive values from upstream nodes. For example:
- A "Split" node auto-fills its `dataset_id` from whichever data source is connected upstream
- An "ML Algorithm" node auto-fills `train_dataset_id` from the Split node
- This means users mostly just need to configure the first node (data source) and tweak a few parameters

### 4.6 Pipeline Execution

When the user clicks "Run":
1. Frontend sends the node graph to the backend
2. Backend sorts nodes in **topological order** (upstream first)
3. Each node executes, passing outputs to downstream nodes
4. Results stream back to the frontend via **SSE (Server-Sent Events)**
5. Each node lights up green (success) or red (error) as it completes
6. User gets **+25 XP** for a successful pipeline run

---

## 5. All Node Types — What Each One Does

### 5.1 Data Source Nodes

| Node | What It Does | How User Configures It |
|------|-------------|----------------------|
| **Upload Dataset** | Upload a CSV file from computer | Click to browse for CSV file |
| **Select Dataset** | Pick from previously uploaded datasets | Dropdown of existing datasets |
| **Sample Dataset** | Load a built-in dataset | Dropdown: Iris, Wine, Breast Cancer, Digits, Titanic, Penguins, Heart Disease, Diabetes, Boston Housing, Tips, Auto MPG, Student Performance, Linnerud |
| **Image Dataset** | Load image datasets or capture with camera/pose | Toggle between: **Builtin** (Digits 8x8, MNIST 28x28, Fashion-MNIST, Shapes 16x16), **Camera** (capture images with webcam), **Pose** (capture body poses with MediaPipe) |

### 5.2 View & Visualization Nodes

| Node | What It Does |
|------|-------------|
| **Table View** | Shows the dataset as an interactive scrollable table |
| **Data Preview** | Shows first N and last N rows (head/tail) |
| **Statistics View** | Mean, std, min, max, quartiles — like pandas `.describe()` |
| **Column Info** | Column data types, missing value counts, unique value counts |
| **Chart View** | Create charts: bar, line, scatter, histogram, pie. User picks x/y columns |

### 5.3 Preprocessing & Feature Engineering Nodes

| Node | What It Does |
|------|-------------|
| **Missing Value Handler** | Handle missing values per column: drop, fill with mean/median/mode, forward/backward fill |
| **Encoding** | Convert categorical columns to numbers. Per-column control (label encoding, one-hot, etc.) |
| **Scaling / Normalization** | Scale numeric features: Standard (z-score), MinMax (0-1), Robust, Normalize |
| **Feature Selection** | Remove low-variance or highly correlated features |

### 5.4 Target & Split Node

| Node | What It Does |
|------|-------------|
| **Target & Split** | Select which column is the target (y), split into train/test sets. Options: train ratio, stratified split, random seed, shuffle |

### 5.5 ML Algorithm Nodes

| Node | Type | Key Hyperparameters |
|------|------|-------------------|
| **Linear Regression** | Regression | fit_intercept |
| **Logistic Regression** | Classification | C (regularization), penalty, solver, max_iter |
| **Decision Tree** | Classification or Regression | max_depth, min_samples_split, min_samples_leaf |
| **Random Forest** | Classification or Regression | n_estimators, max_depth, min_samples_split |
| **MLP Classifier** | Classification | hidden_layer_sizes, activation, solver, learning_rate |
| **MLP Regressor** | Regression | hidden_layer_sizes, activation, solver, learning_rate |
| **K-Means Clustering** | Unsupervised | n_clusters, init_method, max_iter |

### 5.6 Image Pipeline Nodes

The image pipeline is a specialized 3-node flow:

```
Image Dataset → Image Split → Image Predictions
```

| Node | What It Does |
|------|-------------|
| **Image Dataset** | Loads image data. Three sources: **Builtin** (MNIST, Fashion-MNIST, Digits, Shapes), **Camera** (webcam capture), **Pose** (body landmark capture via MediaPipe) |
| **Image Split** | Stratified train/test split. Outputs both `train_dataset_id` and `test_dataset_id` |
| **Image Predictions** | Trains + evaluates in one step. Supports MLP on pixels, MLP on pose landmarks. Shows architecture, training curves, confusion matrix, feature importance, live camera/pose testing |

**Camera Capture flow**: User opens Image Dataset config → selects "Camera" → sees live webcam feed → creates class tabs (e.g., "Cat", "Dog") → captures images per class → clicks "Build Dataset" → dataset saved on server → node configured

**Pose Capture flow**: Same as camera but uses MediaPipe PoseLandmarker → extracts 33 body landmarks (132 features: x, y, z, visibility per landmark) → trains MLP on landmark data

### 5.7 Results & Metric Nodes

| Node | What It Measures | Best For |
|------|-----------------|----------|
| **R² Score** | How well model explains variance (0-1, higher = better) | Regression |
| **MSE** | Mean Squared Error (lower = better) | Regression |
| **RMSE** | Root Mean Squared Error (lower = better) | Regression |
| **MAE** | Mean Absolute Error (lower = better) | Regression |
| **Confusion Matrix** | TP/FP/TN/FN visualization | Classification |

### 5.8 GenAI Nodes

A separate pipeline type for building AI chatbots:

```
LLM Provider → System Prompt → Examples (optional) → Chatbot
```

| Node | What It Does |
|------|-------------|
| **LLM Provider** | Configure which AI model to use: OpenAI, Anthropic Claude, Google Gemini, or DynaRoute (auto-routing). Set temperature, max tokens, API key |
| **System Prompt** | Define the AI's personality/role. Presets: helpful assistant, domain expert, tutor, code reviewer, etc. |
| **Examples (Few-Shot)** | Provide input/output examples so the AI learns the pattern |
| **Chatbot** | Interactive chat interface where the user talks to the configured AI |

---

## 6. Interactive Explorers — What Users See After Running

Every node has an **explorer view** (opened by clicking an executed node). These are rich, interactive panels:

### Data Nodes
- **Gallery**: Thumbnail grid of sample images per class
- **Distribution**: Bar chart of class balance
- **Statistics**: Pixel value histograms
- **Quiz**: Auto-generated multiple-choice questions about the data

### ML Model Nodes
- **Overview**: Accuracy gauge, per-class metrics
- **Architecture**: Visual diagram of the model layers
- **Training Curves**: Loss/accuracy over epochs
- **Confusion Matrix**: Interactive heatmap
- **Feature Importance**: Which features matter most (bar chart or skeleton heatmap for pose)
- **Confidence Distribution**: How confident the model is on each class
- **Live Camera**: Real-time webcam prediction (for image/pose models)
- **Quiz**: Educational questions about the model

### Split Nodes
- **Split Visualization**: Sample images from train vs test sets
- **Class Balance**: Per-class distribution comparison
- **Split Ratios**: Detailed breakdown of train/test sizes

---

## 7. Interactive Activities — Learning Modules

Activities are standalone interactive simulations (no data input needed). They teach ML concepts visually.

### Beginner Level
| Activity | What It Teaches |
|----------|----------------|
| **Loss Functions** | Drag points, see how MSE/MAE/Huber loss changes in real-time |
| **Linear Regression** | Drag a best-fit line, watch error minimize |
| **Gradient Descent** | Visualize step-by-step optimization on a loss surface |
| **Logistic Regression** | Adjust decision threshold on a sigmoid curve |
| **K-Means Clustering** | Drop centroids, watch them converge iteratively |

### Intermediate Level
| Activity | What It Teaches |
|----------|----------------|
| **Decision Tree Builder** | Split data step-by-step, build tree visually |
| **Confusion Matrix** | Adjust threshold, watch TP/FP/TN/FN update dynamically |

### Advanced Level
| Activity | What It Teaches |
|----------|----------------|
| **Activation Functions** | Compare ReLU, Sigmoid, Tanh, Leaky ReLU curves side by side |
| **Neural Network Builder** | Build a network visually, watch data propagate forward |
| **Backpropagation** | Watch gradients flow backward through layers |
| **CNN Filter Visualizer** | Apply convolution filters (edge detect, blur, sharpen) to images |

---

## 8. AI Mentor — Guided Learning

The AI Mentor is a **floating assistant** (bottom-right of the playground) that guides students step-by-step through their first ML pipeline.

### How It Works
1. Student selects an ML algorithm from the sidebar
2. Mentor detects this and starts a **learning flow** — a state machine with 35+ stages
3. At each stage, the mentor gives a suggestion: "Now drag a dataset node onto the canvas"
4. As the student adds/configures nodes, the mentor advances to the next stage
5. After the pipeline runs, the mentor explains the results

### Learning Flow Stages (simplified)
```
Welcome → Algorithm Selected → Add Dataset → Configure Dataset →
Add Column Info → Run Column Info →
(if missing values) Add Missing Value Handler →
(if categorical) Add Encoding →
Add Split → Configure Split → Add Model → Add Metrics →
Run Pipeline → Show Results → Completed
```

### Voice Support
- Mentor can **speak** suggestions using Inworld AI TTS
- Default voice: "Rajat Sir" (voice clone)
- Users can **clone their own voice** from an audio sample
- Three voice modes: Voice First, Text First, Ask Each Time
- Audio is cached on both server and client to avoid duplicate TTS calls

### Mentor Capabilities
- **Greet user** (personalized by time of day)
- **Analyze dataset** (insights, quality issues, suggestions)
- **Analyze pipeline** (suggest next steps, improvements)
- **Explain errors** (human-friendly error explanations with solutions)
- **Model guides** (step-by-step instructions for each algorithm)
- **Dataset guidance** (what data to prepare for each model type)

---

## 9. Gamification System

### XP & Levels
- **20 levels** total (Level 1 = 0 XP → Level 20 = 22,000 XP)
- XP is awarded for actions:

| Action | XP Earned |
|--------|-----------|
| First login | 50 XP |
| Pipeline execution | 25 XP |
| Story completed | 40 XP |
| App published | 35 XP |
| Activity completed | 30 XP |
| Quiz passed | 20 XP |
| Project creation | 15 XP |
| Project shared | 15 XP |
| Dataset uploaded | 10 XP |

- **Level-up animation** plays when student reaches a new level
- **XP toast** shows "+25 XP" after earning

### Badges (10 Total)
| Badge | How to Earn |
|-------|-------------|
| Welcome | Log in for the first time |
| First Pipeline | Execute your first ML pipeline |
| Data Explorer | Use 5 different view nodes |
| Model Master | Train 10 different ML models |
| Perfect Score | Achieve R² > 0.95 on a model |
| Quiz Whiz | Score 100% on 3 quizzes |
| Clean Machine | Use missing values + encoding + scaling in one pipeline |
| App Creator | Publish your first custom app |
| Activity Explorer | Complete all 5 interactive activities |
| Storyteller | Complete your first dataset story |

### Where Users See Gamification
- **Profile page** (`/profile`): Full gamification dashboard — XP progress bar, level badge, all badges
- **Playground**: XP toast appears after pipeline execution
- **Level-up modal**: Celebratory animation on level change

---

## 10. Project Sharing

### How Sharing Works
1. User clicks **Share** button in the playground toolbar
2. System generates a **share token** (unique URL)
3. User gets a link: `https://app.visual-ml.com/shared/:shareToken`
4. Anyone with the link can view the pipeline (no login required)

### Share Permissions
- **Can View**: See the pipeline (always true)
- **Can Clone**: Copy the pipeline to their own account (requires login)
- **Can Edit**: Modify the shared pipeline
- **Can Run**: Execute the pipeline

### Share Statistics
- Owner can see: view count, clone count, when last viewed

---

## 11. Custom App Builder

**URL**: `/app-builder/:appId`

Users can turn their ML pipelines into **standalone web apps** that anyone can use.

### How It Works
1. User creates a pipeline and trains a model
2. Goes to App Builder → selects the pipeline
3. System **auto-suggests UI blocks** based on the pipeline nodes
4. User customizes the UI (drag-and-drop blocks, themes)
5. **Publishes** the app with a custom slug
6. App is accessible at `/app/:slug` — no login required

### What Published Apps Can Do
- Accept user input (form fields based on model features)
- Run the ML pipeline on that input
- Display predictions and results

---

## 12. Backend API — Key Endpoint Groups

All API endpoints are prefixed with `/api/v1/`. Full Swagger docs available at `http://localhost:8000/docs`.

| Group | Prefix | Purpose |
|-------|--------|---------|
| **Auth** | `/auth/` | Student/admin registration, login, Google OAuth, OTP verification, password reset, profile management |
| **Projects** | `/projects/` | CRUD for projects, save/load playground state, sharing |
| **Datasets** | `/datasets/` | Upload CSV, list datasets, preview, download |
| **ML Pipeline** | `/ml/` | Run nodes/pipelines (sync + SSE streaming + async Celery), camera/pose dataset creation, live prediction |
| **GenAI** | `/genai/` | Pipeline/node/edge CRUD, execution, chat with LLMs |
| **Custom Apps** | `/custom-apps/` | Create/publish/execute custom apps |
| **Mentor** | `/mentor/` | Greet, analyze, explain errors, TTS, voice clone, preferences |
| **Gamification** | `/gamification/` | XP awards, level/badge tracking |
| **Tasks** | `/tasks/` | Async task status/results (Celery) |
| **Secrets** | `/genai/secrets/` | Encrypted API key storage for LLM providers |

### Key Endpoints Users Interact With (Indirectly)

| What Happens in UI | Backend Endpoint |
|--------------------|-----------------|
| Run pipeline button | `POST /ml/pipeline/run/stream` (SSE) |
| Save project | `POST /projects/:id/save` |
| Load project | `GET /projects/:id/state` |
| Upload dataset | `POST /datasets/upload` |
| Camera capture → build | `POST /ml/camera/dataset` |
| Pose capture → build | `POST /ml/pose/dataset` |
| Live camera predict | `POST /ml/camera/predict` |
| Live pose predict | `POST /ml/pose/predict` |
| AI mentor speaks | `POST /mentor/generate-speech` |
| Share project | `POST /projects/:id/share` |
| Earn XP | `POST /gamification/award-xp` |

---

## 13. Common User Workflows

### Workflow A: First-Time Student (Guided by Mentor)

1. Sign up → verify email → land on dashboard
2. Click "New Project" → enter project name → opens playground
3. AI Mentor appears: "Welcome! Let's build your first ML pipeline"
4. Mentor guides through: pick algorithm → add dataset → add column info → handle missing values → encode → split → train → add metrics → run
5. Student views results, answers quiz questions, earns XP
6. Student explores interactive activities for deeper understanding

### Workflow B: Image Classification

1. Open playground → drag "Image Dataset" node
2. Configure: select "Builtin" → pick MNIST or Digits
3. Drag "Image Split" → connect → configure test size (20%)
4. Drag "Image Predictions" → connect → configure (MLP, hidden layers, epochs)
5. Run pipeline → view results (accuracy, confusion matrix, architecture, live camera test)

### Workflow C: Pose Detection (New!)

1. Open playground → drag "Image Dataset" node
2. Configure: select "Pose" → PoseCapturePanel opens
3. Create classes (e.g., "Standing", "Sitting", "Waving") → capture poses via webcam
4. MediaPipe extracts 33 body landmarks per frame → build dataset
5. Connect Image Split → Image Predictions → Run
6. View results → open "Live Pose" tab → real-time pose classification with skeleton overlay

### Workflow D: GenAI Chatbot

1. Open playground → click "GenAI Chatbot" template
2. Configure LLM Provider (pick Gemini/OpenAI/Claude, set API key)
3. Configure System Prompt (pick a personality or write custom)
4. Optionally add few-shot examples
5. Run → open Chatbot node → interactive chat with the AI

### Workflow E: Build & Publish an App

1. Build a working pipeline (e.g., Random Forest on Iris dataset)
2. Go to App Builder → select the pipeline
3. Customize UI blocks (input fields, prediction display, theme)
4. Publish with a custom slug (e.g., "iris-classifier")
5. Share link: `/app/iris-classifier` — anyone can use it without login

---

## 14. Development Quick Reference

| Task | Command |
|------|---------|
| Start frontend | `cd client && npm run dev` → `http://localhost:5173` |
| Start backend | `cd server && uvicorn main:app --reload` → `http://localhost:8000` |
| API docs | `http://localhost:8000/docs` (Swagger UI) |
| TypeScript check | `cd client && npx tsc --noEmit` |

### Key Directories

```
Visual-ML/
├── client/                          # React frontend
│   ├── src/
│   │   ├── app/App.tsx              # Router + route definitions
│   │   ├── pages/                   # Page components (dashboard, playground, auth)
│   │   ├── components/playground/   # Canvas, Sidebar, ConfigModal, ViewNodeModal, explorers
│   │   ├── config/
│   │   │   ├── nodeDefinitions.ts   # All node definitions (config fields, icons, categories)
│   │   │   └── templateConfig.ts    # Pre-built pipeline templates
│   │   ├── types/pipeline.ts        # TypeScript types (NodeType union, interfaces)
│   │   ├── features/
│   │   │   ├── gamification/        # XP, levels, badges
│   │   │   └── mentor/             # AI mentor, voice, learning flow
│   │   └── lib/poseLandmarker.ts   # MediaPipe pose detection utility
│   │
├── server/                          # Python FastAPI backend
│   ├── main.py                      # App entry point
│   ├── app/
│   │   ├── api/v1/                  # API route handlers
│   │   ├── ml/
│   │   │   ├── engine.py            # Pipeline execution engine + node registry
│   │   │   └── nodes/              # All ML node implementations (one file per node)
│   │   ├── services/               # Business logic (gamification, S3, etc.)
│   │   ├── models/                 # SQLAlchemy database models
│   │   └── mentor/                 # AI mentor + TTS services
```

---

*For code-level details, see `.claude/PROJECT_SUMMARY.md`.*
