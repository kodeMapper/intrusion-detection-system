/**
 * SparklineChart — Lightweight inline SVG trend line.
 * Applies D3.js skill Pattern B: compute scales in JS, render <polyline> in JSX.
 * No external dependencies.
 */
function SparklineChart({ data = [], color = '#22d3ee', label = '', height = 32, showDot = true }) {
  const width = 200; // viewBox width, actual width is 100%
  const padding = 2;

  if (data.length < 2) {
    return (
      <div className="sparkline-container">
        <div className="sparkline-header">
          <span className="sparkline-label">{label}</span>
          <span className="sparkline-current">---</span>
        </div>
        <svg className="sparkline-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          <line x1={0} y1={height / 2} x2={width} y2={height / 2}
            stroke="var(--border-subtle)" strokeWidth="0.5" strokeDasharray="3,3" />
        </svg>
      </div>
    );
  }

  const currentValue = data[data.length - 1];
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1; // prevent division by zero

  // Scale data to SVG coordinates
  const points = data.map((v, i) => {
    const x = padding + (i / (data.length - 1)) * (width - padding * 2);
    const y = padding + (1 - (v - min) / range) * (height - padding * 2);
    return { x, y };
  });

  const linePoints = points.map(p => `${p.x},${p.y}`).join(' ');

  // Area fill (from line down to bottom)
  const areaPath = `M ${points[0].x},${height} ` +
    points.map(p => `L ${p.x},${p.y}`).join(' ') +
    ` L ${points[points.length - 1].x},${height} Z`;

  const lastPoint = points[points.length - 1];

  // Format display value
  const displayValue = typeof currentValue === 'number'
    ? currentValue >= 1 ? currentValue.toFixed(1) : currentValue.toFixed(3)
    : currentValue;

  return (
    <div className="sparkline-container">
      <div className="sparkline-header">
        <span className="sparkline-label">{label}</span>
        <span className="sparkline-current" style={{ color }}>{displayValue}</span>
      </div>
      <svg className="sparkline-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {/* Area gradient fill */}
        <defs>
          <linearGradient id={`grad-${label.replace(/\s/g,'')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.20" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path
          className="sparkline-area"
          d={areaPath}
          fill={`url(#grad-${label.replace(/\s/g,'')})`}
        />
        {/* Line */}
        <polyline
          className="sparkline-line"
          points={linePoints}
          stroke={color}
        />
        {/* Current value dot */}
        {showDot && (
          <circle
            className="sparkline-dot"
            cx={lastPoint.x}
            cy={lastPoint.y}
            r={2.5}
            fill={color}
            stroke="var(--bg-base)"
            strokeWidth={1}
          />
        )}
      </svg>
    </div>
  );
}

export default SparklineChart;
