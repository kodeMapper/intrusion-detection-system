/**
 * ThreatGauge — Radial SVG gauge showing real-time threat level.
 * Inspired by skill-library product-design (feedback immediat) and D3.js (arc patterns).
 */
function ThreatGauge({ attackRate = 0 }) {
  // attackRate: 0–100 (% of recent alerts that are attacks)
  const size = 140;
  const stroke = 10;
  const radius = (size - stroke) / 2;
  const circumference = Math.PI * radius; // half-circle
  const offset = circumference - (attackRate / 100) * circumference;

  // Threat level thresholds
  let level, color;
  if (attackRate >= 75) {
    level = 'CRITICAL';
    color = '#f43f5e'; // crimson
  } else if (attackRate >= 50) {
    level = 'HIGH';
    color = '#f59e0b'; // amber
  } else if (attackRate >= 25) {
    level = 'ELEVATED';
    color = '#22d3ee'; // cyan
  } else {
    level = 'LOW';
    color = '#10b981'; // neon green
  }

  return (
    <div className="threat-gauge-container">
      <svg
        className="threat-gauge-svg"
        width={size}
        height={size * 0.65}
        viewBox={`0 0 ${size} ${size * 0.65}`}
      >
        {/* Background arc */}
        <path
          className="threat-gauge-bg"
          d={describeArc(size / 2, radius + stroke / 2, radius, 180, 360)}
          strokeWidth={stroke}
        />
        {/* Filled arc */}
        <path
          className="threat-gauge-fill"
          d={describeArc(size / 2, radius + stroke / 2, radius, 180, 360)}
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
        {/* Center value */}
        <text
          x={size / 2}
          y={radius * 0.7}
          textAnchor="middle"
          className="threat-gauge-value"
          fontSize="1.3rem"
          fill={color}
        >
          {Math.round(attackRate)}%
        </text>
        {/* Level label */}
        <text
          x={size / 2}
          y={radius * 0.7 + 16}
          textAnchor="middle"
          className="threat-gauge-level"
          fill={color}
        >
          {level}
        </text>
        {/* Title */}
        <text
          x={size / 2}
          y={size * 0.65 - 2}
          textAnchor="middle"
          className="threat-gauge-label"
        >
          THREAT LEVEL
        </text>
      </svg>
    </div>
  );
}

// Utility: describe a half-circle arc path
function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArc = endAngle - startAngle <= 180 ? '0' : '1';
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return {
    x: cx + r * Math.cos(rad),
    y: cy + r * Math.sin(rad),
  };
}

export default ThreatGauge;
