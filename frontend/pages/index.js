import { useState, useEffect, useRef } from 'react'

const styles = {
  page: { fontFamily: 'Inter, system-ui, -apple-system, Roboto, "Segoe UI", Arial', padding: 20, background: '#f8fafc', minHeight: '100vh' },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 },
  title: { fontSize: 22, fontWeight: 700 },
  caption: { color: '#6b7280' },
  alert: (color) => ({ background: color, color: 'white', padding: 12, borderRadius: 10, fontWeight: 600, textAlign: 'center', boxShadow: `0 0 15px ${color}77`, marginBottom: 8 }),
  cards: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 12 },
  card: { background: 'white', padding: 12, borderRadius: 10, boxShadow: '0 2px 8px rgba(15,23,42,0.06)' },
  chart: { background: 'white', padding: 12, borderRadius: 10, marginTop: 12, height: 140, boxShadow: '0 2px 8px rgba(15,23,42,0.06)' },
  table: { marginTop: 12, borderRadius: 10, overflow: 'hidden', boxShadow: '0 2px 8px rgba(15,23,42,0.06)' },
  tableInner: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '8px 12px', background: '#f1f5f9', color: '#0f172a', fontSize: 13 },
  td: { padding: '8px 12px', background: 'white', color: '#0f172a', fontSize: 13, borderBottom: '1px solid #f1f5f9' },
  smallMuted: { color: '#6b7280', fontSize: 13 },
  explainBox: { background: '#f0f9ff', padding: 10, borderRadius: 8, marginTop: 8, fontSize: 12 },
  badge: { display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, marginRight: 4, marginBottom: 4, background: '#fef3c7', color: '#92400e' }
}

function genEvent() {
  const eventTypes = ['Normal Activity', 'Suspicious Login Detected', 'Anomaly in Network Traffic']
  const weights = [0.8, 0.15, 0.05]
  const r = Math.random()
  let acc = 0
  let chosen = eventTypes[0]
  for (let i = 0; i < eventTypes.length; i++) {
    acc += weights[i]
    if (r <= acc) { chosen = eventTypes[i]; break }
  }
  const severity = chosen.includes('Anomaly') ? 'Critical' : chosen.includes('Suspicious') ? 'Warning' : 'Normal'
  return { timestamp: new Date().toLocaleTimeString(), event: chosen, ip: `192.168.1.${Math.floor(Math.random()*250)+2}`, severity }
}

export default function Home() {
  // IMPORTANT: Start with deterministic state for SSR to avoid hydration mismatches
  // Populate logs on client after mount (via /logs or SSE). Do not generate random events on SSR.
  const [logs, setLogs] = useState([])
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [selectedLog, setSelectedLog] = useState(null)
  
  // Load recent history from backend once on mount
  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const res = await fetch('http://localhost:8000/logs?limit=40')
        const json = await res.json()
        if (mounted && Array.isArray(json.events)) {
          setLogs(json.events.slice(0,40))
          const newAlerts = json.events.filter(e => e.severity !== 'Normal').slice(0, 10)
          setAlerts(newAlerts)
        }
      } catch (err) {
        console.warn('Failed to load logs', err)
      }
    })()
    return () => { mounted = false }
  }, [])

  // Subscribe to backend Server-Sent Events stream with reconnect/backoff
  useEffect(() => {
    let es = null
    let shouldStop = false
    let retryMs = 1000
    const maxRetry = 30000

    const connect = () => {
      if (shouldStop) return
      try {
        es = new EventSource('http://localhost:8000/stream')
      } catch (err) {
        console.warn('EventSource not available', err)
        scheduleReconnect()
        return
      }

      es.onopen = () => {
        console.info('SSE connected')
        retryMs = 1000
      }

      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data)
          // Show analyzing state briefly
          setAnalyzing(true)
          setTimeout(() => setAnalyzing(false), 800)
          
          setLogs((s) => [data, ...s].slice(0, 50))
          
          if (data.severity !== 'Normal') {
            setAlerts((s) => [data, ...s].slice(0, 10))
          }
        } catch (err) {
          console.error('Failed to parse SSE event', err)
        }
      }

      es.onerror = (err) => {
        console.warn('SSE error; closing and scheduling reconnect', err)
        try { es.close() } catch (e) {}
        scheduleReconnect()
      }
    }

    const scheduleReconnect = () => {
      if (shouldStop) return
      setTimeout(() => {
        retryMs = Math.min(maxRetry, Math.floor(retryMs * 1.5))
        connect()
      }, retryMs)
    }

    connect()

    return () => {
      shouldStop = true
      try { es && es.close() } catch (e) {}
    }
  }, [])

  const counts = logs.reduce((acc, l) => { acc[l.severity] = (acc[l.severity]||0)+1; return acc }, {})

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <div style={styles.title}>🛰️ Real-Time Network Anomaly & User Behavior Dashboard</div>
          <div style={styles.caption}>Live log monitoring with AI-based anomaly detection & explainability</div>
        </div>
        <div>
          {analyzing && <span style={{ color: '#0ea5e9', fontWeight: 600, marginRight: 12 }}>⚙️ Analyzing...</span>}
        </div>
      </div>

      {/* Alert Banner */}
      {alerts.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          {alerts.slice(0,3).map((a, i) => (
            <div key={i} style={styles.alert(a.severity === 'Critical' ? '#dc2626' : '#f59e0b')}>
              🚨 {a.event} from {a.src_ip} at {a.timestamp} | Score: {a.score ? a.score.toFixed(3) : 'N/A'}
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', gap: 12 }}>
        {/* Left: Logs Table + Chart */}
        <div style={{ flex: 1 }}>
          <div style={styles.chart}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Event Severity Distribution</div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', height: 80 }}>
              {['Critical','Warning','Normal'].map((k)=> (
                <div key={k} style={{ flex: 1 }}>
                  <div style={{ height: `${(counts[k]||0)/Math.max(1,logs.length)*100}%`, background: k==='Critical'?'#dc2626':k==='Warning'?'#f59e0b':'#10b981', borderRadius: 6 }} />
                  <div style={{ textAlign: 'center', marginTop: 6, fontSize: 13 }}>{k}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={styles.table}>
            <table style={styles.tableInner}>
              <thead>
                <tr>
                  <th style={styles.th}>Timestamp</th>
                  <th style={styles.th}>User</th>
                  <th style={styles.th}>Raw Log</th>
                  <th style={styles.th}>Score</th>
                  <th style={styles.th}>Severity</th>
                  <th style={styles.th}></th>
                </tr>
              </thead>
              <tbody>
                {logs.map((l, idx) => (
                  <tr key={idx} style={{ cursor: 'pointer' }} onClick={() => setSelectedLog(l)}>
                    <td style={styles.td}>{l.timestamp}</td>
                    <td style={styles.td}>{l.user_id || 'N/A'}</td>
                    <td style={{...styles.td, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{l.raw_log || l.event}</td>
                    <td style={styles.td}>{l.score !== null && l.score !== undefined ? l.score.toFixed(3) : '—'}</td>
                    <td style={{...styles.td, fontWeight:600, color: l.severity==='Critical'?'#dc2626': l.severity==='Warning'?'#b45309':'#065f46' }}>{l.severity}</td>
                    <td style={styles.td}>{selectedLog === l ? '👁️' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Stats + Explainability */}
        <div style={{ width: 380 }}>
          <div style={styles.cards}>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Total Logs</div><div style={{ fontSize: 20, fontWeight: 700 }}>{logs.length}</div></div>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Normal</div><div style={{ fontSize: 20, fontWeight: 700 }}>{counts['Normal']||0}</div></div>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Warnings</div><div style={{ fontSize: 20, fontWeight: 700 }}>{counts['Warning']||0}</div></div>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Critical</div><div style={{ fontSize: 20, fontWeight: 700 }}>{counts['Critical']||0}</div></div>
          </div>

          {/* Explainability Panel */}
          {selectedLog && (
            <div style={{ marginTop: 12, ...styles.card }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>🔍 Log Details</div>
              <div style={styles.smallMuted}>Selected: {selectedLog.timestamp}</div>
              <div style={{ fontSize: 13, marginTop: 6 }}>
                <strong>User:</strong> {selectedLog.user_id || 'N/A'} | <strong>Score:</strong> {selectedLog.score !== null && selectedLog.score !== undefined ? selectedLog.score.toFixed(3) : 'N/A'}
              </div>
              <div style={{ fontSize: 12, marginTop: 4, color: '#475569' }}>{selectedLog.raw_log || selectedLog.event}</div>

              {selectedLog.user_behavior && selectedLog.user_behavior.length > 0 && (
                <div style={styles.explainBox}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>👤 User Behavior Flags:</div>
                  {selectedLog.user_behavior.map((ub, i) => (
                    <div key={i} style={styles.badge} className="badge-warning">{ub}</div>
                  ))}
                </div>
              )}

              {selectedLog.top_features && selectedLog.top_features.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>📊 Top Contributing Features:</div>
                  {selectedLog.top_features.map((f, i) => (
                    <div key={i} style={{ fontSize: 11, marginBottom: 2, color: '#475569' }}>
                      <strong>{f.feature}:</strong> {f.value.toFixed(2)} (contrib: {f.contribution.toFixed(3)})
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: 12, ...styles.card }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>System Status</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: analyzing ? '#0ea5e9' : '#10b981' }}>
              {analyzing ? 'Analyzing logs...' : 'Listening for logs'}
            </div>
            <div style={styles.smallMuted}>Backend streaming from network.log</div>
          </div>
        </div>
      </div>
    </div>
  )
}
