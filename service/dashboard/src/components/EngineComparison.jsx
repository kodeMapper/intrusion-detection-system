/**
 * EngineComparison — Side-by-side engine cards with animated bars, AE meter with threshold marker, and verdict.
 * Skill: product-design (visual hierarchy), HIG (clear feedback).
 */
function EngineComparison({ alert, health }) {
  const dlWarmup = health?.dlWarmupRemaining || 0;

  if (!alert) {
    return (
      <div className="glass-panel engine-comparison">
        <div className="section-title">Engine Comparison</div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', textAlign: 'center', padding: '16px 0' }}>
          Waiting for first detection…
        </div>
      </div>
    );
  }

  const mlPred = alert.ml_prediction || '—';
  const mlConf = alert.ml_confidence || 0;
  const dlPred = alert.dl_prediction || '—';
  const dlConf = alert.dl_confidence || 0;
  const s1Prob = alert.dl_stage1_attack_prob || 0;
  const aeScore = alert.ae_anomaly_score || 0;
  const zeroDay = alert.zero_day_flag || false;
  const verdict = alert.prediction || '—';
  const verdictConf = alert.confidence || 0;
  const isWarmup = dlPred === 'warming_up';
  const isNormal = verdict.toLowerCase() === 'normal';

  const threatLevel = verdictConf >= 0.9 ? 'critical'
    : verdictConf >= 0.7 ? 'high'
    : verdictConf >= 0.4 ? 'medium'
    : 'low';

  // AE bar level
  const aeLevel = aeScore >= 0.6 ? 'high' : aeScore >= 0.3 ? 'medium' : 'low';

  return (
    <div className="glass-panel engine-comparison">
      <div className="section-title">Latest Detection — Engine View</div>

      {/* ML Engine */}
      <div className="engine-card ml-accent">
        <div className="engine-card-header">
          <span className="engine-card-name">ML Baseline</span>
          <span className="engine-conf-text">{(mlConf * 100).toFixed(1)}%</span>
        </div>
        <div className="engine-card-pred">{mlPred}</div>
        <div className="engine-bar-track">
          <div className="engine-bar-value ml" style={{ width: `${(mlConf * 100).toFixed(0)}%` }} />
        </div>
      </div>

      {/* DL Engine */}
      <div className={`engine-card dl-accent ${isWarmup ? '' : ''}`}>
        <div className="engine-card-header">
          <span className="engine-card-name">DL Advanced</span>
          {!isWarmup && <span className="engine-conf-text">{(dlConf * 100).toFixed(1)}%</span>}
        </div>
        {isWarmup ? (
          <div className="engine-card-pred warmup">
            Warming Up {dlWarmup > 0 ? `(${dlWarmup} flows)` : ''}
          </div>
        ) : (
          <>
            <div className="engine-card-pred">{dlPred}</div>
            <div className="engine-bar-track">
              <div className="engine-bar-value dl" style={{ width: `${(dlConf * 100).toFixed(0)}%` }} />
            </div>
            <div className="engine-detail-text">
              Stage 1 Prob: {(s1Prob * 100).toFixed(1)}%
            </div>
          </>
        )}
      </div>

      {/* AE Canary */}
      <div className={`engine-card ae-accent ${zeroDay ? 'zd-glow' : ''}`}>
        <div className="engine-card-header">
          <span className="engine-card-name">
            Zero-Day Canary {zeroDay && '⚠'}
          </span>
          <span className="ae-score-value">{aeScore.toFixed(4)}</span>
        </div>
        <div className="ae-meter-container">
          <div className="ae-bar-track">
            <div
              className={`ae-bar-fill ${aeLevel}`}
              style={{ width: `${Math.min(100, aeScore * 100).toFixed(0)}%` }}
            />
            {/* Threshold marker at 60% (the ~0.603 threshold) */}
            <div className="ae-threshold-marker" style={{ left: '60%' }} />
          </div>
        </div>
        {zeroDay && (
          <div className="zd-warning">
            ⚠ Anomalous pattern — possible zero-day
          </div>
        )}
      </div>

      {/* Combined Verdict */}
      <div className={`verdict-card ${!isNormal ? `threat-${threatLevel}` : ''}`}>
        <div className="verdict-title">Combined Verdict</div>
        <div className={`verdict-prediction ${isNormal ? 'normal' : 'attack'}`}>
          {verdict}
        </div>
        <div className="verdict-conf">{(verdictConf * 100).toFixed(1)}% confidence</div>
        <div className="verdict-source">Source: {alert.verdict_source?.toUpperCase()}</div>
      </div>
    </div>
  );
}

export default EngineComparison;
