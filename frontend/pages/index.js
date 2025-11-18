import { useState, useEffect, useRef } from 'react'

const styles = {
  page: { fontFamily: 'Inter, system-ui, -apple-system, Roboto, "Segoe UI", Arial', padding: 20, background: '#f8fafc', minHeight: '100vh' },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 },
  title: { fontSize: 22, fontWeight: 700 },
  caption: { color: '#6b7280' },
  alert: (color) => ({ background: color, color: 'white', padding: 12, borderRadius: 10, fontWeight: 600, textAlign: 'center', boxShadow: `0 0 15px ${color}77` }),
  cards: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 12 },
  card: { background: 'white', padding: 12, borderRadius: 10, boxShadow: '0 2px 8px rgba(15,23,42,0.06)' },
  chart: { background: 'white', padding: 12, borderRadius: 10, marginTop: 12, height: 140, boxShadow: '0 2px 8px rgba(15,23,42,0.06)' },
  table: { marginTop: 12, borderRadius: 10, overflow: 'hidden', boxShadow: '0 2px 8px rgba(15,23,42,0.06)' },
  tableInner: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '8px 12px', background: '#f1f5f9', color: '#0f172a' },
  td: { padding: '8px 12px', background: 'white', color: '#0f172a' },
  smallMuted: { color: '#6b7280', fontSize: 13 }
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
  const [logs, setLogs] = useState([genEvent(), genEvent(), genEvent()])
  const [alert, setAlert] = useState(null)
  const [loading, setLoading] = useState(false)
  const [score, setScore] = useState(null)
  
  // Load recent history from backend once on mount
  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const res = await fetch('http://localhost:8000/logs?limit=40')
        const json = await res.json()
        if (mounted && Array.isArray(json.events)) {
          setLogs(json.events.slice(0,40))
          if (json.events.length && json.events[0].severity !== 'Normal') setAlert(json.events[0])
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
          const mapped = { timestamp: data.timestamp, event: data.event, ip: data.ip, severity: data.severity }
          setLogs((s) => [mapped, ...s].slice(0, 40))
          if (mapped.severity !== 'Normal') setAlert(mapped)
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

  async function pushEvent() {
    const e = genEvent()
    setLogs((s) => [e, ...s].slice(0, 40))
    if (e.severity !== 'Normal') setAlert(e)
    // Call backend for a score (sample payload length matches trained model expectations approx)
    setLoading(true)
    try {
      const payload = { network: Array(16).fill(0).map(() => Math.random()), user: Array(4).fill(0).map(() => Math.random()) }
      const res = await fetch('http://localhost:8000/score', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      const json = await res.json()
      setScore(json.score ?? json)
    } catch (err) {
      setScore({ error: err.message })
    } finally { setLoading(false) }
  }

  const counts = logs.reduce((acc, l) => { acc[l.severity] = (acc[l.severity]||0)+1; return acc }, {})

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <div style={styles.title}>🛰️ Real-Time Network Anomaly & User Behavior Dashboard</div>
          <div style={styles.caption}>Live monitoring with AI-based anomaly alerts</div>
        </div>
        <div>
          <button onClick={pushEvent} style={{ padding: '8px 12px', background: '#0ea5e9', color: 'white', border: 'none', borderRadius: 8 }}>Generate Event</button>
        </div>
      </div>

      {alert ? (
        <div style={{ marginBottom: 12 }}>
          <div style={styles.alert(alert.severity === 'Critical' ? '#dc2626' : '#f59e0b')}>{`🚨 ${alert.event} from ${alert.ip} at ${alert.timestamp}`}</div>
        </div>
      ) : null}

      <div style={{ display: 'flex', gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={styles.chart}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Event Severity</div>
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
                  <th style={styles.th}>Event</th>
                  <th style={styles.th}>Source IP</th>
                  <th style={styles.th}>Severity</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((l, idx) => (
                  <tr key={idx}>
                    <td style={styles.td}>{l.timestamp}</td>
                    <td style={styles.td}>{l.event}</td>
                    <td style={styles.td}>{l.ip}</td>
                    <td style={{...styles.td, fontWeight:600, color: l.severity==='Critical'?'#dc2626': l.severity==='Warning'?'#b45309':'#065f46' }}>{l.severity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ width: 320 }}>
          <div style={styles.cards}>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Total Logs</div><div style={{ fontSize: 20, fontWeight: 700 }}>{logs.length}</div></div>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Normal</div><div style={{ fontSize: 20, fontWeight: 700 }}>{counts['Normal']||0}</div></div>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Warnings</div><div style={{ fontSize: 20, fontWeight: 700 }}>{counts['Warning']||0}</div></div>
            <div style={styles.card}><div style={{ fontSize: 12, color: '#6b7280' }}>Critical</div><div style={{ fontSize: 20, fontWeight: 700 }}>{counts['Critical']||0}</div></div>
          </div>

          <div style={{ marginTop: 12, ...styles.card }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Last Score</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{loading ? 'Calling...' : (score === null ? '—' : (score.error ? 'Error' : Number(score).toFixed(3)))}</div>
            <div style={styles.smallMuted}>Scores are produced by the backend model</div>
          </div>
        </div>
      </div>
    </div>
  )
}
