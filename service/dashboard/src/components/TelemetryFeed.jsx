import { useState } from 'react';

/**
 * TelemetryFeed — Live alert stream with severity-colored rows, muted normals, and expandable details.
 * Skill: ui-pattern (list section), HIG (feedback principle, progressive disclosure).
 */
function TelemetryFeed({ alerts, autoPoll }) {
  const [expandedIdx, setExpandedIdx] = useState(null);

  if (alerts.length === 0) {
    return (
      <div className="glass-panel feed-panel">
        <div className="feed-header">
          <div className="feed-title">
            <span className={`live-dot ${autoPoll ? 'active' : ''}`} />
            Live Telemetry
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-icon">📡</div>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
              No Anomalies Detected
            </div>
            <div style={{ fontSize: '0.75rem' }}>Waiting for incoming telemetry…</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel feed-panel">
      <div className="feed-header">
        <div className="feed-title">
          <span className={`live-dot ${autoPoll ? 'active' : ''}`} />
          Live Telemetry
        </div>
        <span className="feed-count">{alerts.length} alerts</span>
      </div>

      <div className="feed-scroll">
        {alerts.map((alert, idx) => {
          const isNormal = alert.prediction?.toLowerCase() === 'normal';
          const conf = alert.confidence || 0;
          const confPct = (conf * 100).toFixed(1);
          const isExpanded = expandedIdx === idx;
          const hasMultiEngine = alert.ml_prediction !== undefined;

          // Severity class
          let severity = 'normal';
          if (!isNormal) {
            if (conf >= 0.90) severity = 'critical';
            else if (conf >= 0.70) severity = 'high';
            else if (conf >= 0.40) severity = 'medium';
            else severity = 'low';
          }

          const time = alert.detectedAt
            ? new Date(alert.detectedAt).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
            : '—';

          return (
            <div
              key={`${alert.sampleIndex}-${idx}`}
              className={`alert-row severity-${severity}`}
              onClick={() => setExpandedIdx(isExpanded ? null : idx)}
            >
              {/* Index */}
              <div className="alert-idx">#{alert.sampleIndex}</div>

              {/* Body: verdict + inline badges */}
              <div className="alert-body">
                <span className={`alert-verdict ${isNormal ? 'normal' : 'attack'}`}>
                  {alert.prediction}
                </span>

                {hasMultiEngine && !isExpanded && (
                  <>
                    <span className={`badge badge-ml`}>ML:{alert.ml_prediction}</span>
                    <span className={`badge badge-dl ${alert.dl_prediction === 'warming_up' ? 'warmup' : ''}`}>
                      DL:{alert.dl_prediction === 'warming_up' ? 'Warm' : alert.dl_prediction}
                    </span>
                    {alert.ae_anomaly_score !== undefined && (
                      <span className={`badge badge-ae ${aeLevel(alert.ae_anomaly_score)}`}>
                        AE:{alert.ae_anomaly_score.toFixed(2)}
                      </span>
                    )}
                    {alert.zero_day_flag && <span className="badge badge-zd">⚠ ZD</span>}
                  </>
                )}
              </div>

              {/* Meta: confidence + time */}
              <div className="alert-meta">
                <span className="alert-conf">{confPct}%</span>
                <span className="alert-time">{time}</span>
              </div>

              {/* Expanded detail */}
              {isExpanded && hasMultiEngine && (
                <div className="alert-detail">
                  <span className="badge badge-ml">
                    ML: {alert.ml_prediction} ({((alert.ml_confidence || 0) * 100).toFixed(1)}%)
                  </span>
                  <span className={`badge badge-dl ${alert.dl_prediction === 'warming_up' ? 'warmup' : ''}`}>
                    DL: {alert.dl_prediction === 'warming_up' ? 'Warming Up' : alert.dl_prediction}
                    {alert.dl_prediction !== 'warming_up' && ` (${((alert.dl_confidence || 0) * 100).toFixed(1)}%)`}
                  </span>
                  {alert.dl_stage1_attack_prob !== undefined && (
                    <span className="badge badge-source">S1:{(alert.dl_stage1_attack_prob * 100).toFixed(1)}%</span>
                  )}
                  {alert.ae_anomaly_score !== undefined && (
                    <span className={`badge badge-ae ${aeLevel(alert.ae_anomaly_score)}`}>
                      AE: {alert.ae_anomaly_score.toFixed(4)}
                    </span>
                  )}
                  <span className={`badge ${alert.engine_agreement ? 'badge-agree' : 'badge-disagree'}`}>
                    {alert.engine_agreement ? '✓ Agree' : '✗ Disagree'}
                  </span>
                  <span className="badge badge-source">SRC:{alert.verdict_source?.toUpperCase()}</span>
                  {alert.zero_day_flag && <span className="badge badge-zd">⚠ Zero-Day</span>}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function aeLevel(score) {
  if (score >= 0.6) return 'high';
  if (score >= 0.3) return 'medium';
  return 'low';
}

export default TelemetryFeed;
