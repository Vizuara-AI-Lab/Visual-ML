# Visual ML
- We’re building a drag-and-drop ML learning playground where students can visually create end-to-end machine learning workflows:
- **Upload Dataset → Clean Data → Split → Train Model → Evaluate → Share Results → Build Custom UI → Connect Hardware**
- The platform will include **GenAI-powered features** with an integrated **LLM**, allowing users to either **upgrade to advanced models** or **plug in their own API key**.
- To make learning faster, students can start with **ready-made templates**—for example, clicking **Linear Regression** instantly loads all the required nodes and steps for that pipeline.
- We’ll also provide **workflow-style automation** for GenAI tasks, enabling students to generate outputs and run repeated actions with minimal effort.
- Our core differentiators will be:
  - **Custom UI builder for student projects**
  - **Real hardware integration for hands-on learning**
- The platform will include two separate experiences: an **Admin Dashboard** for managing registered users and their data, and a **Student Dashboard** for building, learning, and sharing projects.
- Beyond ML, students will also learn **computational thinking** through block-based logic building, including **conditional statements and decision-making flows**, similar to Scratch-style programming.


Because training inside API request = your server dies.


## Terminal 1: Redis
redis-server
## Terminal 2: Celery Worker
celery -A app.core.celery_app worker --loglevel=info
## Terminal 3: Celery Beat (for scheduled tasks)
celery -A app.core.celery_app beat --loglevel=info
## Terminal 4: FastAPI
uvicorn main:app --reload



{
  "email": "user@example.com",
  "password": "606280Sk$",
  "name": "string"
}

Remove preview_data from the database model (or at least stop storing it)


config model and nodal config model


### 3. N+1 Query Problem Everywhere 💥

**Evidence:**

- File: [server/app/api/v1/projects.py](server/app/api/v1/projects.py#L285-L320)
- File: [server/app/api/v1/genai_pipelines.py](server/app/api/v1/genai_pipelines.py#L117-L140)

**Problem:**
You query nodes and edges separately **AFTER** loading the pipeline:

```python
# Line 312-315 - N+1 QUERIES!
nodes = db.query(GenAINode).filter(GenAINode.pipelineId == project_id).all()  # Query 1
edges = db.query(GenAIEdge).filter(GenAIEdge.pipelineId == project_id).all()  # Query 2
```

**Why It's Slow:**
For 10 projects, this becomes **30 queries** instead of 1.

**Fix - Use SQLAlchemy Eager Loading:**

```python
from sqlalchemy.orm import joinedload

# Single query with joined relationships
project = (
    db.query(GenAIPipeline)
    .options(
        joinedload(GenAIPipeline.nodes),
        joinedload(GenAIPipeline.edges)
    )
    .filter(
        GenAIPipeline.id == project_id,
        GenAIPipeline.studentId == student.id
    )
    .first()
)
```

**Impact:** Reduces **30 queries → 1 query** = **90% faster**.

---


[1] Data Source
   ↓
[2] View / Inspect
   ↓
[3] Preprocessing
   ↓
[4] Feature Engineering
   ↓
[5] Target & Split
   ↓
[6] Linear Regression Model
