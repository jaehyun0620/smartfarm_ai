import { useState, useEffect, useRef, useCallback } from 'react'
import type {
  SensorData, ActuatorState, SystemStatus,
  SensorHistoryPoint, LogEntry, SystemMode, ApiRecord
} from '../types'
import { fetchLatest, fetchHistory, sendCommand, parseRecord, recordToHistoryPoint } from '../api/service'

// ─── Mock fallback (used when API is unreachable) ───────────────────────────

const MOCK_BASE: ApiRecord = {
  timestamp: new Date().toISOString().slice(0, 19),
  overall_state: 'STABLE', environment_state: 'STABLE',
  temp_state: 'NORMAL', humidity_state: 'NORMAL',
  soil_state: 'KEEP', light_state: 'NORMAL',
  co2_state: 'MONITOR_ONLY', day_night: 'DAY',
  temp1: '21.5', hum1: '65.0', temp2: '22.0', hum2: '68.0',
  soil_raw: '575', soil_percent: '50',
  light_raw: '769', water_raw: '815', co2_raw: '325',
  window1_magnetic: 'CLOSED', window2_magnetic: 'CLOSED',
  fan1: '0', fan2: '0', window1: '0', window2: '0',
  heater: '0', humidifier: '0', led: '0', pump: '0',
  window1_command: 'CLOSED', window2_command: 'CLOSED',
  window1_sensor: 'CLOSED', window2_sensor: 'CLOSED',
  scenario: 'LIGHT_NORMAL_LED_OFF',
}

function jitter(val: string, range: number): string {
  return (parseFloat(val) + (Math.random() - 0.5) * range).toFixed(1)
}

function mockNext(prev: ApiRecord): ApiRecord {
  return {
    ...prev,
    timestamp: new Date().toISOString().slice(0, 19),
    temp1: jitter(prev.temp1, 0.8),
    hum1: jitter(prev.hum1, 2),
    temp2: jitter(prev.temp2, 0.5),
    hum2: jitter(prev.hum2, 2),
    soil_percent: String(Math.min(100, Math.max(0, Math.round(parseFloat(prev.soil_percent) + (Math.random() - 0.5) * 2)))),
    light_raw: String(Math.min(1023, Math.max(0, Math.round(parseFloat(prev.light_raw) + (Math.random() - 0.5) * 30)))),
    water_raw: String(Math.min(1023, Math.max(0, Math.round(parseFloat(prev.water_raw) + (Math.random() - 0.5) * 20)))),
    co2_raw: String(Math.min(2000, Math.max(300, Math.round(parseFloat(prev.co2_raw) + (Math.random() - 0.5) * 5)))),
  }
}

// ─── Hook ───────────────────────────────────────────────────────────────────

export function useSensorData() {
  const [mode, setMode] = useState<SystemMode>('auto')
  const [isConnected, setIsConnected] = useState(false)
  const [sensor, setSensor] = useState<SensorData | null>(null)
  const [actuators, setActuators] = useState<ActuatorState[]>([])
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [history, setHistory] = useState<SensorHistoryPoint[]>([])
  const [log, setLog] = useState<LogEntry[]>([])
  const [manualStates, setManualStates] = useState<Record<string, boolean>>({})

  const mockRecordRef = useRef<ApiRecord>(MOCK_BASE)
  const idRef = useRef(0)

  const applyRecord = useCallback((record: ApiRecord) => {
    const { sensor: s, actuators: a, status } = parseRecord(record)

    // In manual mode, override actuator states with user control
    const finalActuators = mode === 'manual'
      ? a.map(act => ({ ...act, isOn: manualStates[act.id] ?? act.isOn }))
      : a

    setSensor(s)
    setActuators(finalActuators)
    setSystemStatus({
      ...status,
      activeActuators: finalActuators.filter(x => x.isOn).length,
    })

    const point = recordToHistoryPoint(record, ++idRef.current)
    setHistory(h => [...h.slice(-19), point])
    setLog(l => [{
      time: point.time,
      temp1: s.temp1,
      hum1: s.hum1,
      soilPercent: s.soilPercent,
      lightRaw: s.lightRaw,
      co2: s.co2,
      waterPercent: s.waterPercent,
    }, ...l.slice(0, 9)])
  }, [mode, manualStates])

  // Initial history load
  useEffect(() => {
    fetchHistory(20)
      .then(records => {
        const pts = records.map((r, i) => recordToHistoryPoint(r, i))
        setHistory(pts)
        idRef.current = pts.length
        setIsConnected(true)
      })
      .catch(() => {
        // Use mock history
        const pts = Array.from({ length: 20 }, (_, i) => {
          const r = mockNext(mockRecordRef.current)
          return recordToHistoryPoint(r, i)
        })
        setHistory(pts)
        idRef.current = 20
      })
  }, [])

  // Polling loop every 2s
  useEffect(() => {
    const tick = async () => {
      try {
        const record = await fetchLatest()
        mockRecordRef.current = record
        setIsConnected(true)
        applyRecord(record)
      } catch {
        setIsConnected(false)
        const next = mockNext(mockRecordRef.current)
        mockRecordRef.current = next
        applyRecord(next)
      }
    }

    tick()
    const id = setInterval(tick, 2000)
    return () => clearInterval(id)
  }, [applyRecord])

  const toggleManual = useCallback((actuatorId: string) => {
    setManualStates(prev => {
      const next = { ...prev, [actuatorId]: !(prev[actuatorId] ?? false) }
      // Fire-and-forget command to API
      sendCommand(actuatorId, next[actuatorId]).catch(() => {})
      return next
    })
  }, [])

  return {
    mode, setMode,
    sensor,
    actuators,
    systemStatus,
    history,
    log,
    isConnected,
    toggleManual,
  }
}
