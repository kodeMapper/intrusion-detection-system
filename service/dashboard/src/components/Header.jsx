import { useState, useEffect } from 'react';

/**
 * Header — Premium brand with gradient text, live clock, and animated connection status.
 * Skill: product-design (Apple-level polish, visual identity).
 */
function Header({ isOnline }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const timeStr = time.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

  return (
    <header className="header-container">
      <div className="brand">
        <div className="brand-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-cyan)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
        </div>
        <div className="brand-text">
          <h1 className="brand-title">Sentinel IDPS</h1>
          <div className="brand-subtitle">Intrusion Detection & Prevention</div>
        </div>
      </div>

      <div className="header-right">
        <span className="header-clock">{timeStr}</span>
        <div className="status-badge">
          <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
          <span>{isOnline ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
