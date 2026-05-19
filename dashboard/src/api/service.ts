import type { ApiRecord, SensorData, ActuatorState, SystemStatus } from '../types'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/** Fetch latest sensor record from FastAPI */
export async function fetchLatest(): Promise<ApiRecord> {
  const res = await fetch(`${API_BASE}/data/latest`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

/** Fetch recent N records for history/log */
export async function fetchHistory(limit = 20): Promise<ApiRecord[]> {
  const res = await fetch(`${API_BASE}/data/history?limit=${limit}`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

/** Send manual actuator command to FastAPI */
export async function sendCommand(actuatorId: string, state: boolean): Promise<void> {
  await fetch(`${API_BASE}/control/${actuatorId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state: state ? 1 : 0 }),
  })
}

// ─── Parsers ────────────────────────────────────────────────────────────────

const ADC_MAX = 1023

export function parseRecord(r: ApiRecord): {
  sensor: SensorData
  actuators: ActuatorState[]
  status: SystemStatus
} {
  const sensor: SensorData = {
    temp1: parseFloat(r.temp1),
    hum1: parseFloat(r.hum1),
    temp2: parseFloat(r.temp2),
    hum2: parseFloat(r.hum2),
    soilPercent: parseFloat(r.soil_percent),
    lightRaw: parseFloat(r.light_raw),
    waterPercent: Math.round((parseFloat(r.water_raw) / ADC_MAX) * 100),
    co2: parseFloat(r.co2_raw),
    timestamp: r.timestamp,
  }

  const actuators: ActuatorState[] = [
    {
      id: 'window1',
      name: '창문 1 (서보)',
      nameEn: 'Window 1 (Servo)',
      icon: '🪟',
      isOn: r.window1 === '1',
      windowState: r.window1_command,
      magneticState: r.window1_magnetic,
      reason: r.window1 === '1' ? '자동 개방 명령' : undefined,
    },
    {
      id: 'window2',
      name: '창문 2 (서보)',
      nameEn: 'Window 2 (Servo)',
      icon: '🪟',
      isOn: r.window2 === '1',
      windowState: r.window2_command,
      magneticState: r.window2_magnetic,
      reason: r.window2 === '1' ? '자동 개방 명령' : undefined,
    },
    {
      id: 'fan1',
      name: '흡기 팬',
      nameEn: 'Fan (Intake)',
      icon: '🌀',
      isOn: r.fan1 === '1',
      reason: r.fan1 === '1' ? '환기 필요' : undefined,
    },
    {
      id: 'fan2',
      name: '배기 팬',
      nameEn: 'Fan (Exhaust)',
      icon: '🌀',
      isOn: r.fan2 === '1',
      reason: r.fan2 === '1' ? '환기 필요' : undefined,
    },
    {
      id: 'heater',
      name: 'PTC 히터',
      nameEn: 'PTC Heater',
      icon: '🔥',
      isOn: r.heater === '1',
      reason: r.heater === '1' ? '온도 낮음' : undefined,
    },
    {
      id: 'humidifier',
      name: '가습기',
      nameEn: 'Humidifier',
      icon: '💨',
      isOn: r.humidifier === '1',
      reason: r.humidifier === '1' ? '습도 낮음' : undefined,
    },
    {
      id: 'led',
      name: '식물 LED',
      nameEn: 'Plant LED',
      icon: '💡',
      isOn: r.led === '1',
      reason: r.led === '1' ? '조도 부족' : undefined,
    },
    {
      id: 'pump',
      name: '관수 펌프',
      nameEn: 'Water Pump',
      icon: '💧',
      isOn: r.pump === '1',
      reason: r.pump === '1' ? '토양수분 부족' : undefined,
    },
  ]

  const activeCount = actuators.filter(a => a.isOn).length

  const status: SystemStatus = {
    overall: r.overall_state,
    tempState: r.temp_state,
    humidityState: r.humidity_state,
    soilState: r.soil_state,
    lightState: r.light_state,
    co2State: r.co2_state,
    dayNight: r.day_night,
    activeActuators: activeCount,
    totalActuators: actuators.length,
    scenario: r.scenario,
  }

  return { sensor, actuators, status }
}

export function recordToHistoryPoint(r: ApiRecord, id: number) {
  return {
    id,
    time: r.timestamp.slice(11, 19),
    temp1: parseFloat(r.temp1),
    hum1: parseFloat(r.hum1),
    soilPercent: parseFloat(r.soil_percent),
    lightRaw: parseFloat(r.light_raw),
    co2: parseFloat(r.co2_raw),
    waterPercent: Math.round((parseFloat(r.water_raw) / 1023) * 100),
  }
}
