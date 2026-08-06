/**
 * ControlPanel — Premium toggle switch and ingestion button.
 * Skill: product-design (iOS-style toggle, clean interactions).
 */
function ControlPanel({ isApiOnline, autoPoll, setAutoPoll, onManualPoll, isManualPolling }) {
  return (
    <div className="glass-panel controls-section">
      <div className="section-title">Ingestion Controls</div>

      {/* Auto-Sync Toggle */}
      <div className="control-row">
        <span className="control-label">Live Auto-Sync</span>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={autoPoll}
            onChange={(e) => setAutoPoll(e.target.checked)}
            disabled={!isApiOnline}
          />
          <span className="toggle-track" />
          <span className="toggle-thumb" />
        </label>
      </div>

      {/* Manual Poll Button */}
      <button
        className="btn btn-primary"
        onClick={onManualPoll}
        disabled={!isApiOnline || isManualPolling || autoPoll}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="23 4 23 10 17 10" />
          <polyline points="1 20 1 14 7 14" />
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
        </svg>
        {isManualPolling ? 'Ingesting…' : 'Ingest 1 Sample'}
      </button>

      {!isApiOnline && (
        <div className="offline-banner">
          API offline — start Node server on port 3001
        </div>
      )}
    </div>
  );
}

export default ControlPanel;
