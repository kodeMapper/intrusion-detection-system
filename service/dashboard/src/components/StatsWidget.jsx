import ThreatGauge from './ThreatGauge';
import SparklineChart from './SparklineChart';

/**
 * StatsWidget (KPI Strip) — Compact left-panel stats with threat gauge, sparklines, and engine breakdown.
 * Skill: ui-pattern (stat grid), product-design (premium feel), HIG (progressive disclosure).
 */
function StatsWidget({ health, alerts, stats, alertHistory }) {
  const processed = health?.processedSamples || 0;
  const mode = health?.mode || 'unknown';
  const dlWarmup = health?.dlWarmupRemaining || 0;

  // Compute attack rate from recent history
  const recentAlerts = alertHistory || [];
  const recentAttacks = recentAlerts.filter(a => a.prediction?.toLowerCase() !== 'normal').length;
  const attackRate = recentAlerts.length > 0 ? (recentAttacks / recentAlerts.length) * 100 : 0;

  // Critical alerts count
  const countCritical = alerts.filter(a => a.confidence >= 0.9).length;

  // Stats from API
  const mlOnly = stats?.mlOnlyDetections || 0;
  const dlOnly = stats?.dlOnlyDetections || 0;
  const bothDet = stats?.bothDetections || 0;
  const zeroDayCount = stats?.zeroDay || 0;
  const agreementRate = stats?.engineAgreementRate || 'N/A';

  // Sparkline data
  const confidenceHistory = recentAlerts.map(a => a.confidence || 0);
  const aeHistory = recentAlerts.map(a => a.ae_anomaly_score || 0);

  return (
    <>
      {/* Threat Gauge */}
      <div className="glass-panel" style={{ padding: '8px' }}>
        <ThreatGauge attackRate={attackRate} />
      </div>

      {/* Mode + Warmup */}
      <div className="kpi-card">
        <div className="kpi-label">
          <span className="kpi-label-icon">⚡</span> Engine Mode
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
          <span className={`mode-chip ${mode === 'csv_fallback' ? 'csv' : 'mongo'}`}>
            {mode === 'csv_fallback' ? 'CSV' : 'MongoDB'}
          </span>
          {dlWarmup > 0 && (
            <span className="warmup-chip">DL Warming ({dlWarmup})</span>
          )}
        </div>
      </div>

      {/* Processed samples */}
      <div className="kpi-card">
        <div className="kpi-label">
          <span className="kpi-label-icon">📊</span> Processed
        </div>
        <div className="kpi-value">{processed.toLocaleString()}</div>
      </div>

      {/* Critical alerts */}
      <div className="kpi-card">
        <div className="kpi-label">
          <span className="kpi-label-icon">🔴</span> Critical Alerts
        </div>
        <div className={`kpi-value ${countCritical > 0 ? 'danger' : ''}`}>
          {countCritical}
        </div>
        <div className="kpi-sub">Confidence ≥ 90%</div>
      </div>

      {/* Zero-Day Flags */}
      <div className={`kpi-card ${zeroDayCount > 0 ? 'zd-active' : ''}`}>
        <div className="kpi-label">
          <span className="kpi-label-icon">⚠</span> Zero-Day Flags
        </div>
        <div className={`kpi-value ${zeroDayCount > 0 ? 'warn' : ''}`}>
          {zeroDayCount}
        </div>
      </div>

      {/* Engine Detection Breakdown */}
      <div className="kpi-card">
        <div className="kpi-label">
          <span className="kpi-label-icon">🔍</span> Detection Sources
        </div>
        <div className="engine-breakdown" style={{ marginTop: '4px' }}>
          <div className="engine-row">
            <span className="engine-dot ml" />
            <span className="engine-name">ML Only</span>
            <span className="engine-count">{mlOnly}</span>
          </div>
          <div className="engine-row">
            <span className="engine-dot dl" />
            <span className="engine-name">DL Only</span>
            <span className="engine-count">{dlOnly}</span>
          </div>
          <div className="engine-row">
            <span className="engine-dot both" />
            <span className="engine-name">Both Agreed</span>
            <span className="engine-count">{bothDet}</span>
          </div>
        </div>
      </div>

      {/* Agreement Rate */}
      <div className="kpi-card">
        <div className="kpi-label">
          <span className="kpi-label-icon">🤝</span> Engine Agreement
        </div>
        <div className="kpi-value info" style={{ fontSize: '1.3rem' }}>{agreementRate}</div>
      </div>

      {/* Sparklines */}
      <div className="glass-panel" style={{ padding: '10px 14px' }}>
        <SparklineChart
          data={confidenceHistory}
          color="#22d3ee"
          label="Confidence Trend"
          height={28}
        />
      </div>

      <div className="glass-panel" style={{ padding: '10px 14px' }}>
        <SparklineChart
          data={aeHistory}
          color="#f59e0b"
          label="AE Anomaly Score"
          height={28}
        />
      </div>
    </>
  );
}

export default StatsWidget;
