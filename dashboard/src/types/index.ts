export type SystemMode = 'auto' | 'manual'

export type StateLevel = 'NORMAL' | 'HIGH' | 'LOW'
export type OverallState = 'STABLE' | 'ACTIVE'
export type SoilState = 'KEEP' | 'ENOUGH'
export type WindowState = 'OPEN' | 'CLOSED'
export type DayNight = 'DAY' | 'NIGHT'

/** Raw JSON shape returned by FastAPI */
export interface ApiRecord {
  timestamp: string
  overall_state: OverallState
  environment_state: string
  temp_state: StateLevel
  humidity_state: StateLevel
  soil_state: SoilState
  light_state: StateLevel
  co2_state: string
  day_night: DayNight
  // Sensor values (strings from API)
  temp1: string
  hum1: string
  temp2: string
  hum2: string
  soil_raw: string
  soil_percent: string
  light_raw: string
  water_raw: string
  co2_raw: string
  // Window magnetic sensors
  window1_magnetic: WindowState
  window2_magnetic: WindowState
  // Actuator states ("0" | "1")
  fan1: string
  fan2: string
  window1: string
  window2: string
  heater: string
  humidifier: string
  led: string
  pump: string
  // Window commands and sensors
  window1_command: WindowState
  window2_command: WindowState
  window1_sensor: WindowState
  window2_sensor: WindowState
  scenario: string
}

/** Parsed, display-ready sensor data */
export interface SensorData {
  temp1: number
  hum1: number
  temp2: number
  hum2: number
  soilPercent: number
  lightRaw: number
  waterPercent: number
  co2: number
  timestamp: string
}

/** Parsed actuator state */
export interface ActuatorState {
  id: string
  name: string
  nameEn: string
  icon: string
  isOn: boolean
  reason?: string
  // For windows: extra state info
  windowState?: WindowState
  magneticState?: WindowState
}

export interface SystemStatus {
  overall: OverallState
  tempState: StateLevel
  humidityState: StateLevel
  soilState: SoilState
  lightState: StateLevel
  co2State: string
  dayNight: DayNight
  activeActuators: number
  totalActuators: number
  scenario: string
}

export interface SensorHistoryPoint {
  id: number
  time: string
  temp1: number
  hum1: number
  soilPercent: number
  lightRaw: number
  co2: number
  waterPercent: number
}

export interface LogEntry {
  time: string
  temp1: number
  hum1: number
  soilPercent: number
  lightRaw: number
  co2: number
  waterPercent: number
}
