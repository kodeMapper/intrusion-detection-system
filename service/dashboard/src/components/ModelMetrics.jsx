/**
 * ModelMetrics — Compact metric cards with color-coded values.
 * Skill: ui-pattern (stat grid, detail card), core-components (tokens).
 */
function ModelMetrics({ metrics }) {
  if (!metrics) {
    return (
      <div className="glass-panel metrics-section">
        <div className="section-title">Model Evaluation Metrics</div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', padding: '12px 0', textAlign: 'center' }}>
          Loading metrics…
        </div>
      </div>
    );
  }

  const s1 = metrics.stage1_threshold;
  const s1Report = metrics.stage1_binary;
  const s2Report = metrics.stage2_multiclass;
  const aeThresh = metrics.ae_threshold;

  return (
    <div className="glass-panel metrics-section">
      <div className="section-title">Model Metrics (Train / Val / Test)</div>

      {/* Stage 1 Binary */}
      <MetricCard
        title="DL Stage 1 — Binary Gate"
        dotColor="var(--accent-purple)"
        items={buildS1Items(s1, s1Report)}
      />

      {/* Stage 2 Multiclass */}
      <MetricCard
        title="DL Stage 2 — Multiclass"
        dotColor="var(--accent-purple)"
        items={buildS2Items(s2Report)}
      />

      {/* AE Canary */}
      <MetricCard
        title="AE — Zero-Day Sensor"
        dotColor="var(--accent-amber)"
        items={buildAEItems(aeThresh)}
      />
    </div>
  );
}

function MetricCard({ title, dotColor, items }) {
  return (
    <div className="metric-card">
      <div className="metric-card-title">
        <span className="metric-dot" style={{ background: dotColor }} />
        {title}
      </div>
      <div className="metric-grid">
        {items.map((item, i) => (
          <div key={i} className="metric-item">
            <span className="metric-item-label">{item.label}</span>
            <span className={`metric-item-value ${metricColorClass(item.value, item.isPercent)}`}>
              {item.display}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function metricColorClass(value, isPercent) {
  if (value === null || value === undefined) return 'na';
  const v = isPercent ? value : value * 100;
  if (v >= 90) return 'good';
  if (v >= 70) return 'ok';
  return 'poor';
}

function fmt(v, asPercent = true) {
  if (v === null || v === undefined) return { display: '—', value: null, isPercent: false };
  const display = asPercent ? `${(v * 100).toFixed(1)}%` : v.toFixed(4);
  return { display, value: v, isPercent: asPercent };
}

function buildS1Items(s1, s1Report) {
  const items = [];
  const valMetrics = s1?.threshold_selection?.validation_metrics;

  if (valMetrics) {
    items.push({ label: 'Attack Recall (Val)', ...fmt(valMetrics.attack_recall) });
    items.push({ label: 'Normal Recall (Val)', ...fmt(valMetrics.normal_recall) });
    items.push({ label: 'F1 Score', ...fmt(valMetrics.f1, false) });
    items.push({ label: 'Threshold', display: s1?.attack_probability_threshold?.toFixed(4) || '—', value: null, isPercent: false });
  }

  if (s1Report?.test) {
    items.push({ label: 'Accuracy (Test)', ...fmt(s1Report.test.accuracy) });
    items.push({ label: 'Attack Recall (Test)', ...fmt(s1Report.test.attack_recall) });
  }

  if (items.length === 0) {
    items.push({ label: 'Status', display: 'No data', value: null, isPercent: false });
  }

  return items;
}

function buildS2Items(s2Report) {
  const items = [];

  if (s2Report?.validation?.argmax) {
    const v = s2Report.validation.argmax;
    const t = s2Report.test?.argmax;

    if (v.macro_recall !== undefined) items.push({ label: 'Macro Recall (Val)', ...fmt(v.macro_recall) });
    if (v.macro_f1 !== undefined) items.push({ label: 'Macro F1 (Val)', ...fmt(v.macro_f1, false) });
    if (v.accuracy !== undefined) items.push({ label: 'Accuracy (Val)', ...fmt(v.accuracy) });
    if (t?.macro_recall !== undefined) items.push({ label: 'Macro Recall (Test)', ...fmt(t.macro_recall) });
  }

  if (items.length === 0) {
    items.push({ label: 'Status', display: 'No data', value: null, isPercent: false });
  }

  return items;
}

function buildAEItems(aeThresh) {
  const items = [];

  if (aeThresh) {
    items.push({ label: 'Threshold', display: aeThresh.E_thresh?.toFixed(4) || '—', value: null, isPercent: false });
    items.push({ label: 'Score Method', display: aeThresh.score_method || '—', value: null, isPercent: false });

    const valMetrics = aeThresh.threshold_selection?.validation_metrics;
    if (valMetrics) {
      items.push({ label: 'Normal Recall', ...fmt(valMetrics.normal_recall) });
      items.push({ label: 'Attack Recall', ...fmt(valMetrics.attack_recall) });
    }
  }

  if (items.length === 0) {
    items.push({ label: 'Status', display: 'No data', value: null, isPercent: false });
  }

  return items;
}

export default ModelMetrics;
