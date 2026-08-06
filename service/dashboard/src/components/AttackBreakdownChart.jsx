/**
 * AttackBreakdownChart — Horizontal bar chart of attack category distribution.
 * Follows the D3.js skill's bar chart pattern adapted to pure React SVG.
 */

const ATTACK_COLORS = {
  DoS:       '#f43f5e',
  Exploits:  '#f59e0b',
  Fuzzers:   '#a78bfa',
  Generic:   '#22d3ee',
  Reconnaissance: '#2dd4bf',
  Normal:    '#3d4d65',
  // fallback
  Other:     '#5a6a82',
};

function AttackBreakdownChart({ breakdown = {} }) {
  const entries = Object.entries(breakdown)
    .filter(([k]) => k.toLowerCase() !== 'normal')
    .sort((a, b) => b[1] - a[1]);

  if (entries.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '12px 16px' }}>
        <div className="section-title" style={{ marginBottom: '8px' }}>Attack Breakdown</div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', textAlign: 'center', padding: '12px 0' }}>
          No attack data yet
        </div>
      </div>
    );
  }

  const maxCount = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <div className="glass-panel" style={{ padding: '12px 16px' }}>
      <div className="section-title" style={{ marginBottom: '8px' }}>Attack Breakdown</div>
      <div className="breakdown-chart">
        {entries.map(([category, count]) => {
          const color = ATTACK_COLORS[category] || ATTACK_COLORS.Other;
          const pct = (count / maxCount) * 100;
          return (
            <div key={category} className="breakdown-row">
              <span className="breakdown-label" title={category}>{category}</span>
              <div className="breakdown-bar-bg">
                <div
                  className="breakdown-bar-fill"
                  style={{ width: `${pct}%`, background: color }}
                />
              </div>
              <span className="breakdown-count">{count}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default AttackBreakdownChart;
