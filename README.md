# 📞 LeadScore Router

**Real-time ML system that scores incoming call center leads and routes them to the optimal agent queue.**

Built with FastAPI, scikit-learn, and Streamlit.

---

## What It Does

Call centers waste senior agent time on low-value callers. This system:

1. Receives caller data (duration, IVR choice, wait time, etc.)
2. Scores conversion likelihood 0–100 using a trained Random Forest model
3. Routes the caller to one of three queues:
   - **HIGH_VALUE_QUEUE** (score ≥ 70) → senior agents
   - **STANDARD_QUEUE** (40 ≤ score < 70) → regular agents
   - **SELF_SERVICE** (score < 40) → automated options

Every prediction is logged for auditing and drift detection.

---

## Tech Stack

- **Python 3.13**
- **scikit-learn** — Random Forest classifier (~78% accuracy)
- **FastAPI** — REST API with Pydantic validation
- **Uvicorn** — ASGI server
- **Streamlit** — Interactive web UI
- **Pandas / NumPy** — Data processing

---

## How to Run Locally

### 1. Clone & setup

```bash
git clone https://github.com/YOUR_USERNAME/leadscore-router.git
cd leadscore-router
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt