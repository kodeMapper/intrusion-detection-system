# KodeMapper — AI-Driven Intrusion Detection & Prevention System (IDPS)

An AI-powered network Intrusion Detection and Prevention System with explainable alerts (SHAP/LIME), automated remediation (SOAR-lite), and a real-time monitoring dashboard. Built as a final-year project.

**Notion Docs:** [link](https://www.notion.so/AI-Based-Intrusion-Detection-System-Linux-ML-2fcee24f567980e7ae04c4ad697a7fb9?source=copy_link)

---

## Key Features

- **Hybrid Multi-Engine Detection:** Runs a unified pipeline combining an ML Baseline (XGBoost, Random Forest, LightGBM), a DL Advanced Engine (Temporal Transformers for sequence modeling), and an AE Anomaly Canary (TemporalOneClassVAE for zero-day detection).
- **Explainable alerts (primary innovation):** Per-alert SHAP/LIME explanations visualized in the dashboard so operators understand *why* traffic was flagged.
- **SOAR-lite automated remediation (secondary innovation):** Maps alerts to safe mitigation actions (iptables block, quarantine) with human-approval gates and one-click rollback.
- **Real-time dashboard:** React-based UI with alert timeline, SHAP force plots, and SOAR controls.
- **Multi-channel alerting:** Email (SMTP), Slack webhook, and push notifications.
- **Multi-dataset evaluation:** NSL-KDD, CICIDS2017, UNSW-NB15 with cross-dataset generalization tests.
- **Safe demo mode:** Prevention actions are simulated by default — no risk of disrupting networks.

---

## Repository Structure

```
/
├── documentation/          # All project documentation (10 docs)
│   ├── 01_project_overview.md
│   ├── 02_tech_and_architecture.md
│   ├── 03_implementation_plan.md
│   ├── 04_monthly_schedule_6months.md
│   ├── 05_experimental_plan_and_metrics.md
│   ├── 06_deployment_and_ops.md
│   ├── 07_api_and_user_manual.md
│   ├── 08_references_and_attributions.md
│   ├── 09_test_plan_and_checklist.md
│   └── 10_future_work_and_risks.md
├── data/                   # Raw/processed datasets (or download scripts)
├── notebooks/              # Jupyter notebooks (EDA, experiments)
├── service/                # All runtime components (actual convention; supersedes the src/ sketch below)
│   ├── collector/          # Pcap/log collectors, parsers (tcpdump, Zeek)
│   ├── preproc/            # Feature extraction, encoding, scaling
│   ├── models/             # ML/DL training scripts, saved models
│   ├── api/                # Existing Node/Express demo bridge (Mongo-backed)
│   ├── detection_api/      # FastAPI backend: detection, explainability (SHAP/LIME), alerting
│   ├── dashboard/          # React frontend
│   └── automation/         # SOAR-lite: iptables/Suricata automation
├── infra/                  # Dockerfiles, docker-compose, Kubernetes manifests
├── experiments/            # Training logs, metrics, MLflow artifacts
├── tests/                  # Unit, integration, performance tests
├── literature/             # Literature survey and research papers
└── globalPrompt.txt        # Project specification
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| OS | Ubuntu 22.04 LTS |
| Data Collection | tcpdump, tshark, Zeek |
| ML/DL | Python, scikit-learn, XGBoost, LightGBM, PyTorch (Transformers/VAE) |
| Explainability | SHAP, LIME |
| Backend | FastAPI, PostgreSQL, Redis |
| Dashboard | React, Chart.js, D3.js |
| Alerting | SMTP, Slack Webhook, Firebase |
| Prevention | iptables, Suricata (demo mode) |
| Monitoring | Prometheus, Grafana |
| CI/CD | GitHub Actions, Docker |

---

## Quick Start

### Prerequisites

- Docker 24.0+ and Docker Compose v2.20+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/<org>/idps-project.git
cd idps-project

# Copy and configure environment
cp .env.example .env
# Edit .env with your database password, JWT secret, etc.

# Build and start all services
docker compose -f infra/docker-compose.yml up -d

# Initialize database
docker compose exec api python -m src.api.db.migrate
docker compose exec api python -m src.api.db.seed
```

### Access

| Service | URL |
|---------|-----|
| Dashboard (Sentinel) | http://localhost:5173 |
| API (Swagger) | http://localhost:8000/docs |
| Grafana | http://localhost:3001 |
| Prometheus | http://localhost:9090 |

### Download Datasets

```bash
bash data/scripts/download_datasets.sh
```

### Train Models

```bash
python -m src.models.train --dataset cicids2017 --model xgboost --seed 42
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## Documentation

All project documentation is in the `documentation/` folder:

1. **[Project Overview](documentation/01_project_overview.md)** — Problem statement, scope, innovations, threat model
2. **[Tech & Architecture](documentation/02_tech_and_architecture.md)** — Architecture diagrams, tech stack, protocols
3. **[Implementation Plan](documentation/03_implementation_plan.md)** — Modules, classes, APIs, data schemas
4. **[Monthly Schedule](documentation/04_monthly_schedule_6months.md)** — 6-month timeline with milestones
5. **[Experimental Plan](documentation/05_experimental_plan_and_metrics.md)** — Datasets, models, metrics, baselines
6. **[Deployment & Ops](documentation/06_deployment_and_ops.md)** — Docker configs, monitoring, rollback
7. **[API & User Manual](documentation/07_api_and_user_manual.md)** — API spec, dashboard guide, demo script
8. **[References](documentation/08_references_and_attributions.md)** — Paper attributions, dataset and tool citations
9. **[Test Plan](documentation/09_test_plan_and_checklist.md)** — Test cases, checklists, coverage targets
10. **[Future Work & Risks](documentation/10_future_work_and_risks.md)** — Limitations, extensions, risk register

---

## Team

**KodeMapper** — Final Year Project Team (4 members)

| Member | Role |
|--------|------|
| A | Data Engineering & ML Pipeline |
| B | Infrastructure & DevOps |
| C | Backend API & Integration |
| D | Frontend Dashboard & Documentation |

---

## License

This project is for academic purposes. All referenced research papers and datasets are cited in [documentation/08_references_and_attributions.md](documentation/08_references_and_attributions.md).