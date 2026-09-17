# Dev 1 Progress Tracker — Explainability, Detection API & ML Integration

Branch: `explainability` (cut from `dl-training-phase`)
Plan reference: see conversation / `dev_assignment_plan.md` (local, gitignored) for the full assignment and contracts.
Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/needs decision

---

## Phase 0 — Spike (dependency install + compatibility checks) — DONE
- [x] Install shap, lime, fastapi, uvicorn, pydantic, pydantic-settings, httpx, pytest, pytest-asyncio, pytest-cov into `.venv`
- [x] Verify shap/lime import cleanly under numpy 2.4 / pandas 3.0 / Python 3.11
- [x] Time `shap.TreeExplainer` on one preprocessed ML-baseline row
- [x] Try `shap.GradientExplainer` on a `(1,10,198)` window through the Transformer Stage1 model
- [x] Record findings below

**Findings:**
- Installed cleanly: shap 0.51.0, lime 0.2.0.1, fastapi 0.141.1, pydantic 2.13.5, httpx 0.28.1, pytest 9.1.1 — no conflicts with existing numpy 2.4.6/pandas 3.0.5/torch 2.13.0/python 3.11.9.
- **Confirmed Finding 0** empirically: `feature_selector.pkl` has no `feature_names_in_` attribute.
- **New finding, resolved with user input:** `shap.TreeExplainer(model_output="probability")` raises `Exception: Model does not have a known objective or output type!` for the XGBoost model specifically — shap 0.51's internal objective map has no entry for XGBoost's multiclass `multi:softprob` objective (RF and LightGBM both work fine in probability mode). Decision (user-confirmed): use `model_output="raw"` uniformly for all 3 models instead of probability+background. Empirically verified RF's "raw" output already equals exact probability (`base_value + Σshap == predict_proba`, floating-point exact); LightGBM/XGBoost "raw" is log-odds/margin space. Per-model additivity is asserted in each model's own output space; the cross-model average is used for ranking/display only, not claimed as an exact unified-probability decomposition.
- Warm per-request cost (explainers built once at startup, as designed): xgb ~111ms, rf ~751ms, lgbm ~42ms → ~900ms total for all 3 tree explainers combined. Comfortably under the 5s budget.
- `shap.GradientExplainer` on `TemporalTransformerClassifier` (Stage1): works cleanly, 0.117s for `nsamples=50`, output shape `(1,10,198,2)` as expected. No issues — validates skipping straight to GradientExplainer.
- `shap.DeepExplainer` on the same model: **fails as anticipated** ("SHAP explanations do not sum up to the model's output," unrecognized `LayerNorm` op) — confirms the design decision to never attempt DeepExplainer and go straight to GradientExplainer, falling back to LIME only if GradientExplainer itself fails on a given checkpoint.
- LIME (`lime_tabular`, regression mode, single-last-timestep perturbation): works cleanly, 0.028s for 200 samples — validates the "freeze first 9 timesteps, perturb only the most recent" design for the AE/DL fallback explainer.

---

## Phase 1 — Scaffolding + test infra
- [ ] `pytest.ini` (root)
- [ ] `tests/conftest.py` (path bootstrap, `artifacts_available` fixture, shared fixtures)
- [ ] `service/detection_api/src/__init__.py` + subpackage `__init__.py`s (routes, services, models)

## Phase 2 — `service/models/src/explain.py` (pure logic, no models) — DONE
- [x] `tests/unit/test_explain.py` written (RED — confirmed `ModuleNotFoundError` before implementation)
- [x] Value types + helpers implemented (GREEN): `ExplanationMethod`, `FeatureAttribution`, `ExplanationResult`, `direction_of`, `rank_attributions`, `aggregate_sequence`, `group_one_hot`
- [x] `tests/unit/test_explanation_fallback.py` written (RED)
- [x] `safe_explain` implemented (GREEN)
- Result: 25/25 tests passing (`pytest tests/unit/test_explain.py tests/unit/test_explanation_fallback.py`)

## Phase 3 — `service/detection_api/src/models/schemas.py` — DONE
- [x] Frozen contract types (`TopFeature`, `ShapExplanation`, `PredictionResponse`) + snapshot test
- [x] `FlowInput`, `DetectRequest`, `DetectResponse`
- [x] `ModelStatus`/`ModelPerformance`/`InferenceStats`/`WindowState`/`ModelsStatusResponse`
- [x] Provisional `AlertCreate`/`Alert`/`AlertFilter`/`AlertPage`/`AlertRepository` Protocol + `derive_severity()` table
- Result: `tests/unit/test_schemas.py` 14/14 passing. Full suite so far: 39/39 passing.

## Phase 4 — Alert repository + alerting — DONE
- [x] `service/detection_api/src/services/alert_repository.py` (`InMemoryAlertRepository`, provisional) + `tests/unit/test_alert_repository.py` (7 tests, not in the plan's named list but added since this is new business logic — pagination, locking, filtering)
- [x] `tests/unit/test_alerting_email.py` written (RED) — 9 tests
- [x] `tests/unit/test_alerting_slack.py` written (RED) — 7 tests
- [x] `service/detection_api/src/services/alerting.py` implemented (GREEN): `EmailNotifier`, `SlackNotifier`, `CompositeNotifier`, `NullNotifier`, `AlertingSettings`
- **Finding (fixed, no user input needed — universally correct fix):** httpx logs the full request URL at INFO level by default, which would leak the Slack webhook's embedded secret path into application logs. Fixed by suppressing httpx's own request-logging (`logging.getLogger("httpx").setLevel(logging.WARNING)`) in `alerting.py`.
- Result: 67/67 tests passing across the whole suite so far.

## Phase 5 — `service/detection_api/src/services/detection.py` — DONE
- [x] Additive `DLAdvancedEngine.predict_from_tensor()` added to `unified_predictor_worker.py` — behavior-preservation verified empirically (`predict()` and `predict_from_tensor()` on the same buffered window produce byte-identical output against real artifacts)
- [x] `tests/unit/test_flow_window_store.py` written (RED) — 10 tests covering cold/warm transition, key isolation, LRU eviction, TTL expiry, stats
- [x] `tests/unit/test_detection_service.py` written (RED) — 14 tests covering the adapter mapping, cold/warm window handling, model-name mapping, explanation attach/failure, persist-only-for-attacks-or-zero-day, `DetectionUnavailableError` on repo failure, persist-before-notify ordering
- [x] `FlowWindowStore`, adapter logic, `DetectionService`, `DetectionUnavailableError`, `DetectionConfig` implemented (GREEN)
- Note: `UnifiedEngineBundle` (the concrete in-process wrapper around the real `MLBaselineEngine`/`DLAdvancedEngine`/`AEAnomalyEngine`) is deferred to Phase 7/8 alongside the real explainers, since `DetectionService` depends only on the `EngineBundle` Protocol — no artifact-loading code needed yet to reach 100% of Phase 5's own test coverage.
- Result: 24/24 new tests passing. Full suite so far: 91/91 passing.

## Phase 6 — Routes — DONE
- [x] `tests/api/test_detect_endpoint.py` written (RED) — 15 tests: 422 validation, 200 happy/degraded, 503 engine-unavailable, flow-key derivation, Finding-0 regression guard
- [x] `service/detection_api/src/routes/detect.py` implemented (GREEN) — fixed a real DI bug along the way: the route must take `service=Depends(get_detection_service)` as a parameter, not call `get_detection_service(request)` manually in the body, or `app.dependency_overrides` silently has no effect
- [x] `tests/api/test_models_status_endpoint.py` written (RED) — 3 tests including a basename-only artifact-path guard (no absolute filesystem paths from an unauthenticated endpoint)
- [x] `service/detection_api/src/routes/models.py` implemented (GREEN) via an overridable `ModelsStatusProvider` Protocol; a real provider backed by actual artifact metadata is deferred to Phase 7/8
- Fixture fix: `tests/conftest.py`'s `_row_as_sample` now converts pandas/numpy scalars (`numpy.int64` etc.) to plain Python types — needed once fixtures started going through `TestClient`'s JSON encoder
- Result: 18 new tests. Full suite so far: 109/109 passing.

## Phase 7 — Real explainers — mostly DONE (dispatcher deferred)
- [x] `tests/unit/test_shap_tree_explainer.py` + `TreeEnsembleExplainer` — 4/4 passing against real `final_xgb/rf/lgbm.pkl`, additivity holds exactly per-model in raw output space
- [x] `tests/unit/test_shap_dl_explainer.py` + `DeepSequenceExplainer` — 3/3 passing against the real Stage1 checkpoint using real preprocessed windows (not random noise)
- [x] `tests/unit/test_lime_ae_explainer.py` + `LimeSequenceExplainer` — 3/3 passing against the real `TemporalOneClassVAE` checkpoint
- [ ] `ExplanationService` dispatcher (routes by `verdict_source`) — deferred; not required to pass any currently-listed test, and wiring it needs the real `UnifiedEngineBundle` (Phase 8/integration territory) to have something meaningful to dispatch between. `DetectionService` already accepts a plain `explain_fn` callable so the dispatcher can be added later without changing its interface.

**Two real bugs found and fixed during this phase (not user-facing yet since nothing else calls this code path yet, but worth flagging for the record):**
1. **Test bug (mine):** `group_one_hot()` was already designed (Phase 2) to collapse the 156 one-hot `proto_*`/`service_*`/`state_*` columns into 3 logical features before ranking — so `DeepSequenceExplainer`/`LimeSequenceExplainer` correctly return 45 feature names (42 numeric/derived + 3 grouped), not 198. My first draft of `test_shap_dl_explainer.py`/`test_lime_ae_explainer.py` asserted 198, contradicting my own already-agreed design. Fixed the test expectations, not the code.
2. **Real implementation bug:** `DeepSequenceExplainer.explain()` originally computed `predicted_value`/`base_value` via `model.predict_proba(...)` (post-softmax probabilities), but `shap.GradientExplainer` computes gradients of the model's raw `forward()` output (pre-softmax logits) — confirmed by inspecting `shap.GradientExplainer`'s API (no `expected_value` attribute, unlike `TreeExplainer`) and by direct debugging: reconstructed-vs-actual additivity error was ~1.0–1.4 (an obviously wrong scale mismatch, not noise) and got *worse*, not better, as `nsamples` increased — the signature of comparing two different unit spaces, not sampling variance. Fixed by computing both values from the model's raw logits (`self._model(window)`), matching what SHAP actually attributes; residual additivity error dropped to ~0.26 at `nsamples=100` with a fixed seed, which is the explainer's genuine (and now correctly small) Monte Carlo approximation error. Bumped `nsamples` default 50→100 for slightly better fidelity (still ~0.1s per call, far under the 5s budget).
- Result: 10 new tests. Full suite so far: 119/119 passing.

## Phase 8 — Integration test
- [ ] `tests/integration/test_predict_pipeline.py`

## Phase 9 — Hardening (partial: review + HIGH fixes done; full Phase 9 closes with Phase 8 wiring)
- [x] `pytest --cov` ≥ 80% on new code — **95%** overall (`service/detection_api/src` 95%, `explain.py` 99%); `detection.py` 100%
- [x] code-reviewer run on diff — 2 HIGH, 3 MEDIUM, 3 LOW found; both HIGH fixed (below)
- [x] security-reviewer run on `detect.py` + `alerting.py` (scope widened to `schemas.py`/`detection.py`/`explain.py` since they're the same trust boundary) — 2 HIGH, 4 MEDIUM, 2 LOW found; both HIGH documented as Phase 8 blockers (below), 2 MEDIUM fixed now
- [ ] `requirements.txt` additions appended (add-only) — done in Phase 0/1, unchanged since

**Fixed as a result of this review (124/124 tests passing, up from 119 — 5 new regression tests added):**
1. **Real bug (code-reviewer HIGH):** `TopFeature.value` collapsed to the SHAP value itself for ~93% of features returned by `DeepSequenceExplainer`/`LimeSequenceExplainer` — `group_one_hot`'s `value_map` only ever populated the 3 grouped categorical names (`proto`/`service`/`state`), so every passthrough numeric feature fell through `rank_attributions`' `transformed_values` fallback, which was passed the *same array* as the SHAP values being ranked. Fixed in `explain.py` by merging `raw_sample` into the `raw_values` dict passed to `rank_attributions` in both explainers, so numeric features report their real value. Regression tests added in `test_shap_dl_explainer.py` and `test_lime_ae_explainer.py`.
2. **Real bug (code-reviewer HIGH):** `attack_probability` in `detection.py` was keyed off `window is not None` instead of which engine actually produced the verdict — in a disagreement (e.g. ML drives an attack verdict while DL's warm window says Normal), the response returned `prediction="DoS", confidence=0.95` alongside a contradicting `attack_probability` sourced from DL's low, disagreeing number. Fixed by branching on `verdict["verdict_source"]` instead. Two regression tests added (`TestAttackProbabilitySource`).
3. **Security (MEDIUM, security-reviewer):** internal exception text (e.g. a future DB driver error with connection strings/hostnames) was forwarded verbatim to the HTTP caller via `DetectionUnavailableError(f"failed to persist alert: {exc}")` → 503 `detail`. Fixed to log the real exception server-side only and raise a generic `"detection service temporarily unavailable"` message.
4. **Resilience (LOW, security-reviewer):** a notifier exception was unhandled and would turn an already-successfully-persisted detection into an HTTP 500. Fixed with a narrow `try/except Exception` around the notify call (logged, not swallowed silently — the alert is already safely persisted either way). Regression test added.
5. **Security (MEDIUM, security-reviewer):** `FlowInput`'s float fields (`dur`, `rate`, `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `tcprtt`, `synack`, `ackdat`) accepted JSON `Infinity`/`-Infinity` (Pydantic v2 defaults `allow_inf_nan=True`; `Field(ge=0)` doesn't reject `+inf`). Fixed with `allow_inf_nan=False` on all 11 fields.
6. **Security (LOW, security-reviewer):** `flow_id` was the one unbounded string field in `FlowInput` (every sibling identifier field is capped). Fixed with `max_length=128`, matching `flow_key`.

**Documented as known limitations, not fixed now — require a Phase 8 wiring decision or are out of Dev 1's scope:**
1. **[HIGH, security]** `DetectionService.detect_flow` calls the (synchronous) ML/DL/SHAP/LIME engine methods inline on the event loop with no thread-pool offload. Harmless today because `EngineBundle` is only a `Protocol` exercised by fast in-memory fakes in tests — but the moment Phase 8 wires in the real `UnifiedEngineBundle` (genuinely CPU-expensive PyTorch/XGBoost/SHAP/LIME calls), a single request can stall the whole event loop for every concurrent caller. **Action for Phase 8:** wrap the engine/explainer calls in `anyio.to_thread.run_sync` (matching the pattern `EmailNotifier` already uses) with bounded concurrency across a batch, not sequential `await`s.
2. **[HIGH, security]** `_derive_flow_key`'s `"__global__"` fallback (when a caller omits both `flow_key` and `src_ip`/`dst_ip`) puts unrelated callers' flows into one shared sliding window, and a caller-supplied `flow_key` is trusted with no ownership check — a caller could target another flow's window by guessing its `src_ip->dst_ip`-shaped key. Not fixed now because a real fix (reject anonymous flows, or bind `flow_key` to authenticated caller identity) depends on Dev 2's auth layer, which doesn't exist yet. **Action for Phase 8/integration:** revisit once JWT auth lands — `flow_key` derivation should incorporate authenticated caller/tenant identity, not trust client-supplied values outright.
3. **[MEDIUM]** `FlowWindowStore` has no internal lock. Safe today only because nothing `await`s between its sync calls in `detect_flow`; becomes unsafe the moment #1 above introduces thread-pool offload for the same store. **Action for Phase 8:** add an `asyncio.Lock` (or move to a lock-free structure) when the real engine bundle is wired in.
4. **[MEDIUM]** Two disconnected `min_notify_severity` settings exist — `AlertingSettings.min_notify_severity` (env-configurable) has no effect on actual gating, since `DetectionService._persist_and_notify` reads only `DetectionConfig.min_notify_severity`. **Action for Phase 8:** collapse to one source of truth when the app assembly point (`main.py`) is built.
5. **[LOW]** `AlertingSettings.suppression_window_s` is defined but never read — no debounce exists yet, so a persistently-triggering flow fires a fresh notification every detection. Flagged, not implemented (YAGNI until it's actually needed).
6. **[LOW]** `InMemoryAlertRepository` and `FlowWindowStore`'s "unbounded growth" concern for the repository specifically: it's explicitly provisional (Dev 2 owns the real Postgres-backed version) — acceptable for local/demo use, flagged so it isn't left running unbounded in any shared environment before Dev 2's implementation lands.
7. **[LOW]** `_PROTOCOL_NUMBERS` only maps tcp/udp/icmp; other UNSW-NB15 protocols persist as `0` in `AlertCreate` silently. Low priority, no test currently exercises other protocols.
8. **[MEDIUM, self-found during DoD cross-check, not flagged by either review agent]** The implementation plan's Phase 4 design explicitly called for notifications to be "invoked via FastAPI BackgroundTasks so SMTP/Slack latency never blocks the HTTP response." The actual code does not do this — `DetectionService._persist_and_notify` `await`s `self._notifier.send(...)` inline, so it's part of the same request/response cycle as `/detect`. Today this is a performance concern rather than a correctness one (the notifier's own timeouts/retries — up to ~2s for SMTP, up to 10s worst-case for Slack's 5s timeout × 2 attempts — are added to the caller's response latency for every attack/zero-day flow), but it's a real deviation from the documented design. Grouped with #1 above (both are "offload blocking work off the request path") as a Phase 8 follow-up: thread the route's `BackgroundTasks` through to the service, or have `detect_flow` return once persisted and schedule notify separately.
8. **[Coverage gaps reviewed, judged acceptable]:** `schemas.py:246,253-255` (uncovered `derive_severity` branches — the table is simple and correct by inspection, just not every branch has a dedicated test); `detect.py:27,35` and `models.py:29` (defensive `HTTPException` paths that need `app.state` left unset, exercised structurally but not via the exact missed line); `alert_repository.py:41,43` (minor pagination edge lines); `alerting.py` (87%, lowest of any file — the missed lines are `NullNotifier`/`CompositeNotifier`'s less-common branches and a few SMTP/Slack permanent-failure paths; behavior is straightforward and was manually reasoned through during the security review, not blocking). `explain.py:106,213` are defensive fallbacks (`rank_attributions`' final "no raw or transformed value available" branch, and `_class_slice`'s single-class-output branch) not hit by current fixtures. None of these represent known-wrong behavior; noted here rather than chased to 100% given diminishing returns.

**Not re-flagged (already correctly mitigated per both reviewers):** httpx webhook-URL logging leak (Phase 4), `FlowWindowStore`'s LRU+TTL bound enforcement (correct), `SecretStr` usage, Slack SSRF prefix guard, unseen-categorical-value handling in the ML preprocessor, email header injection (all values are enum-like/config, never free text), `ExplanationResult.failure_reason` correctly dropped before reaching the API response.

---

## Decisions log
- Model integration: in-process import (not subprocess) — see plan.
- Persistence: attacks/zero-days only; 503 on repo failure.
- SHAP tree mode: probability + background sample.
- New service dir: `service/detection_api/` (not root `src/`) — `dev_assignment_plan.md` and `README.md` updated accordingly.
- `InMemoryAlertRepository` authored by Dev 1 as a provisional stand-in (normally Dev 2's deliverable).

## Open items / questions raised to user
_(none yet — will log here as they come up)_
