import { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import StatsWidget from './components/StatsWidget';
import ControlPanel from './components/ControlPanel';
import TelemetryFeed from './components/TelemetryFeed';
import EngineComparison from './components/EngineComparison';
import ModelMetrics from './components/ModelMetrics';
import AttackBreakdownChart from './components/AttackBreakdownChart';

const API_URL = 'http://127.0.0.1:3001';
const MAX_HISTORY = 50;

function App() {
  const [alerts, setAlerts] = useState([]);
  const [healthData, setHealthData] = useState(null);
  const [statsData, setStatsData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [isApiOnline, setIsApiOnline] = useState(false);
  const [autoPoll, setAutoPoll] = useState(false);
  const [isManualPolling, setIsManualPolling] = useState(false);

  // Rolling history for sparklines (last N alerts)
  const alertHistoryRef = useRef([]);
  const [alertHistory, setAlertHistory] = useState([]);

  // Fetch /health
  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        setHealthData(await res.json());
        setIsApiOnline(true);
      } else {
        setIsApiOnline(false);
      }
    } catch {
      setIsApiOnline(false);
      setHealthData(null);
    }
  }, []);

  // Fetch /alerts
  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/alerts`);
      if (res.ok) {
        const data = await res.json();
        const reversed = data.alerts ? [...data.alerts].reverse() : [];
        setAlerts(reversed);

        // Update rolling history with new alerts
        if (reversed.length > 0) {
          const latest = reversed[0];
          const history = alertHistoryRef.current;
          // Only add if this is genuinely a new alert
          if (history.length === 0 || history[history.length - 1].sampleIndex !== latest.sampleIndex) {
            history.push(latest);
            if (history.length > MAX_HISTORY) history.shift();
            alertHistoryRef.current = history;
            setAlertHistory([...history]);
          }
        }
      }
    } catch (err) {
      console.error("Failed to fetch alerts", err);
    }
  }, []);

  // Fetch /stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/stats`);
      if (res.ok) setStatsData(await res.json());
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  }, []);

  // Fetch /metrics (once)
  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/metrics`);
      if (res.ok) setMetricsData(await res.json());
    } catch (err) {
      console.error("Failed to fetch metrics", err);
    }
  }, []);

  // Manual poll
  const handleManualPoll = async () => {
    setIsManualPolling(true);
    try {
      const res = await fetch(`${API_URL}/poll-once`, { method: 'POST' });
      if (res.ok) {
        await Promise.all([fetchHealth(), fetchAlerts(), fetchStats()]);
      }
    } catch (err) {
      console.error("Failed manual poll", err);
    } finally {
      setIsManualPolling(false);
    }
  };

  // Sync loop
  useEffect(() => {
    fetchHealth();
    fetchAlerts();
    fetchStats();
    fetchMetrics();

    const interval = setInterval(() => {
      fetchHealth();
      fetchStats();
      if (autoPoll) {
        fetchAlerts();
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [autoPoll, fetchHealth, fetchAlerts, fetchStats, fetchMetrics]);

  // Latest alert for engine comparison
  const latestAlert = alerts.length > 0 ? alerts[0] : null;

  // Compute attack breakdown from stats
  const attackBreakdown = statsData?.attackBreakdown || {};

  return (
    <div className="app-container">
      <Header isOnline={isApiOnline} />

      <main className="dashboard-grid">
        {/* ─── LEFT: KPI Strip ─────────────────────────────── */}
        <div className="left-strip">
          <StatsWidget
            health={healthData}
            alerts={alerts}
            stats={statsData}
            alertHistory={alertHistory}
          />
        </div>

        {/* ─── CENTER: Live Feed + Controls + Breakdown ────── */}
        <div className="center-main">
          <div style={{ display: 'flex', gap: 'var(--sp-md)', alignItems: 'stretch' }}>
            <div style={{ flex: 1 }}>
              <ControlPanel
                isApiOnline={isApiOnline}
                autoPoll={autoPoll}
                setAutoPoll={setAutoPoll}
                onManualPoll={handleManualPoll}
                isManualPolling={isManualPolling}
              />
            </div>
            <div style={{ flex: 1 }}>
              <AttackBreakdownChart breakdown={attackBreakdown} />
            </div>
          </div>
          <TelemetryFeed alerts={alerts} autoPoll={autoPoll} />
        </div>

        {/* ─── RIGHT: Engine Comparison + Metrics ──────────── */}
        <div className="right-panel">
          <EngineComparison alert={latestAlert} health={healthData} />
          <ModelMetrics metrics={metricsData} />
        </div>
      </main>
    </div>
  );
}

export default App;
