import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'

// ── API 설정 ──────────────────────────────────────────────────────────────────
// 백엔드 연결 시 여기만 수정하면 됨
const API_BASE_URL = 'https://smelting-evolution-grumpily.ngrok-free.dev'
const API_ENDPOINT = `${API_BASE_URL}/api/sensor-logs`   // GET → records[]
const POLL_INTERVAL_MS = 30_000                           // 30초마다 자동 갱신

// ── State → UI 매핑 ───────────────────────────────────────────────────────────
const STATE_COLOR = {
  NORMAL:       'green',
  STABLE:       'green',
  KEEP:         'green',
  MONITOR_ONLY: 'green',
  HIGH:         'red',
  CRITICAL:     'red',
  LOW:          'orange',
  WARNING:      'orange',
  DRY:          'orange',
}

const STATE_LABEL = {
  NORMAL:       '정상',
  STABLE:       '정상',
  KEEP:         '정상',
  MONITOR_ONLY: '정상',
  HIGH:         '높음',
  CRITICAL:     '위험',
  LOW:          '부족',
  WARNING:      '주의',
  DRY:          '건조',
}

// ── 데이터 변환 ───────────────────────────────────────────────────────────────
// records: 백엔드에서 받은 배열 (오래된 순 정렬)
// 반환: { sensors, statusBars, deviceStates, chartData, logRows, plantSummary, latest }
function transformRecords(records) {
  if (!records || records.length === 0) return null

  const latest = records[records.length - 1]

  // 중첩 구조(sensors/actuators/states) 또는 flat 구조 모두 지원
  const s  = latest.sensors   ?? latest   // 센서값
  const a  = latest.actuators ?? latest   // 구동부
  const st = latest.states    ?? latest   // 상태

  // 센서 값 계산
  const temp     = parseFloat(s.temp1)
  const hum      = parseFloat(s.hum1)
  const soil     = parseInt(s.soil_percent)
  const lightRaw = parseInt(s.light_raw)
  const co2      = parseInt(s.co2_raw)
  const waterPct = Math.min(100, Math.round(parseInt(s.water_raw) / 1023 * 100))

  // ── 센서 카드 데이터 ──
  const sensors = {
    soil:    { value: soil,    unit: '%',   label: '토양 수분', range: '적정 40~70%',      status: soil >= 40 && soil <= 70 ? '정상' : soil < 40 ? '부족' : '과습',       color: soil >= 40 && soil <= 70 ? 'green' : soil < 40 ? 'orange' : 'blue' },
    co2:     { value: co2,     unit: 'ppm', label: 'CO₂',      range: '적정 400~1000ppm', status: STATE_LABEL[st.co2]       ?? '정상', color: STATE_COLOR[st.co2]       ?? 'green' },
    temp:    { value: temp,    unit: '°C',  label: '온도',      range: '적정 18~28°C',     status: STATE_LABEL[st.temp]      ?? '정상', color: STATE_COLOR[st.temp]      ?? 'green' },
    humidity:{ value: hum,     unit: '%',   label: '습도',      range: '적정 50~70%',      status: STATE_LABEL[st.humidity]  ?? '정상', color: STATE_COLOR[st.humidity]  ?? 'green' },
    light:   { value: lightRaw,unit: 'lx',  label: '조도',      range: '적정 500~2000lx',  status: STATE_LABEL[st.light]     ?? '정상', color: STATE_COLOR[st.light]     ?? 'green' },
    water:   { value: waterPct,unit: '%',   label: '수위',      range: '주의 20% 이하',    status: waterPct > 20 ? '충분' : '부족',                                        color: waterPct > 20 ? 'blue' : 'red' },
  }

  // ── 홈 상태바 ──
  const co2Fill  = co2 < 800 ? 80 : co2 < 1200 ? 40 : 20
  const co2Color = co2 < 800 ? 'green' : co2 < 1200 ? 'orange' : 'red'
  const lightFill = Math.min(95, Math.round(lightRaw / 20))
  const thOk     = st.temp === 'NORMAL' && st.humidity === 'NORMAL'

  const statusBars = [
    {
      key: 'soil', icon: '💧', label: '수분',
      fill: soil, color: soil >= 40 && soil <= 70 ? 'green' : soil < 40 ? 'orange' : 'blue',
      text: soil >= 40 && soil <= 70 ? '좋아요' : soil < 40 ? '부족해요' : '과습해요',
      detail: soil < 40
        ? '토양 수분이 부족해요. 물을 주세요.'
        : soil > 70
        ? '토양이 과습 상태예요. 물 주기를 멈춰주세요.'
        : '토양 수분이 적정 범위에 있어요. 현재 상태를 유지해 주세요.',
    },
    {
      key: 'air', icon: '☴', label: '공기',
      fill: co2Fill, color: co2Color,
      text: co2 < 800 ? '맑아요' : co2 < 1200 ? '주의해요' : '탁해요',
      detail: co2 < 800
        ? '공기가 맑아요. 환기 상태가 좋아요.'
        : co2 < 1200
        ? 'CO₂ 농도가 높아지고 있어요. 환기를 해주세요.'
        : 'CO₂ 농도가 높아요. AI가 환기팬을 작동시켰어요. 창문을 열면 더 빨리 개선돼요.',
    },
    {
      key: 'sun', icon: '☀️', label: '햇빛',
      fill: lightFill, color: STATE_COLOR[st.light] ?? 'green',
      text: st.light === 'NORMAL' ? '딱 좋아요' : st.light === 'LOW' ? '어두워요' : '너무 밝아요',
      detail: st.light === 'NORMAL'
        ? '조도가 적정 범위에 있어요. 현재 상태를 유지해 주세요.'
        : st.light === 'LOW'
        ? '조도가 부족해요. LED를 켜거나 채광이 좋은 곳으로 옮겨주세요.'
        : '조도가 높아요. 직사광선을 피해주세요.',
    },
    {
      key: 'th', icon: '🌡', label: '온도·습도',
      fill: thOk ? 80 : 45, color: thOk ? 'green' : 'orange',
      text: thOk ? '쾌적해요' : '주의해요',
      detail: thOk
        ? '온도와 습도가 적정 범위에 있어요. AI가 관리하고 있어요.'
        : `온도 ${st.temp === 'NORMAL' ? '정상' : st.temp} / 습도 ${st.humidity === 'NORMAL' ? '정상' : st.humidity} — 자동으로 조절 중이에요.`,
    },
    {
      key: 'water', icon: '≋', label: '물통 수위',
      fill: waterPct, color: 'blue',
      text: waterPct > 50 ? '충분해요' : waterPct > 20 ? '절반 남았어요' : '부족해요',
      detail: waterPct > 50
        ? '물통에 물이 충분해요.'
        : waterPct > 20
        ? '물통 물이 절반 정도 남았어요. 여유 있을 때 보충해 주세요.'
        : '물통이 거의 비었어요. 물을 채워주세요.',
    },
  ]

  // ── 기기 상태 ──
  const isOn = (v) => v === '1' || v === 1
  const deviceStates = [
    isOn(a.fan1),
    isOn(a.fan2),
    isOn(a.window1),
    isOn(a.window2),
    isOn(a.led),
    isOn(a.humidifier),
    isOn(a.heater),
  ]

  // ── 차트 데이터 ──
  const toTime = (ts) => {
    const d = new Date(ts)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  }
  const getSensor = (r, key) => { const s = r.sensors ?? r; return s[key] }
  const chartData = {
    soil:     records.map(r => ({ t: toTime(r.timestamp), v: parseInt(getSensor(r, 'soil_percent')) })),
    co2:      records.map(r => ({ t: toTime(r.timestamp), v: parseInt(getSensor(r, 'co2_raw')) })),
    temp:     records.map(r => ({ t: toTime(r.timestamp), v: parseFloat(getSensor(r, 'temp1')) })),
    humidity: records.map(r => ({ t: toTime(r.timestamp), v: parseFloat(getSensor(r, 'hum1')) })),
    light:    records.map(r => ({ t: toTime(r.timestamp), v: parseInt(getSensor(r, 'light_raw')) })),
    water:    records.map(r => ({ t: toTime(r.timestamp), v: Math.min(100, Math.round(parseInt(getSensor(r, 'water_raw')) / 1023 * 100)) })),
  }

  // ── 센서 기록 테이블 (최근 10개, 최신순) ──
  const logRows = [...records].reverse().slice(0, 10).map(r => ({
    time:     toTime(r.timestamp),
    temp:     `${parseFloat(getSensor(r, 'temp1')).toFixed(1)}°C`,
    hum:      `${parseFloat(getSensor(r, 'hum1')).toFixed(0)}%`,
    soil:     `${getSensor(r, 'soil_percent')}%`,
    soilNum:  parseInt(getSensor(r, 'soil_percent')),
    co2:      parseInt(getSensor(r, 'co2_raw')),
    light:    parseInt(getSensor(r, 'light_raw')),
    water:    `${Math.min(100, Math.round(parseInt(getSensor(r, 'water_raw')) / 1023 * 100))}%`,
  }))

  // ── 홈 식물 요약 ──
  const issues = []
  if (soil < 40)         issues.push('물이 부족해요')
  if (co2 > 1000)        issues.push('공기가 탁해요')
  if (st.temp     !== 'NORMAL') issues.push('온도가 불안정해요')
  if (st.humidity !== 'NORMAL') issues.push('습도를 조절해야 해요')

  const plantSummary = {
    title:    issues.length === 0 ? '상태가 양호해요' : issues.length === 1 ? '일부 항목을 확인해 주세요' : '여러 항목에서 주의가 필요해요',
    subtitle: issues.length === 0 ? '모든 상태가 좋아요' : issues.join(', '),
  }

  return { sensors, statusBars, deviceStates, chartData, logRows, plantSummary, latest }
}

// ── Mock 데이터 (백엔드 연결 전 fallback) ─────────────────────────────────────
const MOCK_RECORDS = Array.from({ length: 10 }, (_, i) => {
  const d = new Date('2026-05-15T15:58:50')
  d.setMinutes(d.getMinutes() - (9 - i) * 2)
  return {
    timestamp:        d.toISOString().slice(0, 19),
    overall_state:    'WARNING',
    temp_state:       'NORMAL',
    humidity_state:   'NORMAL',
    soil_state:       'LOW',
    light_state:      'NORMAL',
    co2_state:        'HIGH',
    temp1:            (22 + i * 0.1).toFixed(1),
    hum1:             (61 + i * 0.1).toFixed(1),
    temp2:            (24 + i * 0.1).toFixed(1),
    hum2:             (63 + i * 0.1).toFixed(1),
    soil_percent:     String(28 + i * 2),
    light_raw:        String(820 + i * 4),
    water_raw:        '815',
    co2_raw:          String(1420 - i * 40),
    fan1:             '1', fan2: '1',
    window1:          '0', window2: '0',
    led:              '1', humidifier: '0', heater: '0', pump: '0',
    window1_magnetic: 'CLOSED', window2_magnetic: 'CLOSED',
    window1_command:  'CLOSED', window2_command:  'CLOSED',
    window1_sensor:   'CLOSED', window2_sensor:   'CLOSED',
    scenario:         'CO2_HIGH_VENTILATION',
  }
})

// ── 탭 / 기기 정의 ────────────────────────────────────────────────────────────
const TABS = [
  { key: 'soil',     label: '토양 수분' },
  { key: 'co2',      label: 'CO₂'      },
  { key: 'temp',     label: '온도'      },
  { key: 'humidity', label: '습도'      },
  { key: 'light',    label: '조도'      },
  { key: 'water',    label: '수위'      },
]

const DEVICE_META = [
  { icon: '☴',  name: 'fan 1',    sub: '흡기', type: 'on-off',    autoDescOn: '신선한 공기를 안으로 들이는 중', autoDescOff: '지금은 환기가 필요 없어요' },
  { icon: '☴',  name: 'fan 2',    sub: '배기', type: 'on-off',    autoDescOn: '탁한 공기를 밖으로 내보내는 중', autoDescOff: '지금은 환기가 필요 없어요' },
  { icon: '▬',  name: 'window 1', sub: '앞면', type: 'open-close',autoDescOn: '환기를 위해 열어뒀어요',          autoDescOff: '지금은 닫아두는 게 더 나아요' },
  { icon: '▬',  name: 'window 2', sub: '뒷면', type: 'open-close',autoDescOn: '환기를 위해 열어뒀어요',          autoDescOff: '지금은 닫아두는 게 더 나아요' },
  { icon: '💡', name: 'grow_led', sub: '',     type: 'on-off',    autoDescOn: '잘 자랄 수 있게 빛을 켜줬어요',   autoDescOff: '지금은 자연광으로 충분해요' },
  { icon: '☁',  name: 'humid',    sub: '',     type: 'on-off',    autoDescOn: '습도를 높이는 중이에요',           autoDescOff: '지금 습도가 딱 좋아서 쉬는 중이에요' },
  { icon: '🔥', name: 'heater',   sub: '',     type: 'on-off',    autoDescOn: '온도를 높이는 중이에요',           autoDescOff: '지금 온도가 딱 좋아서 쉬는 중이에요' },
]

// ── 색상 팔레트 ───────────────────────────────────────────────────────────────
const COLOR = {
  orange: { text: '#d97706', bar: '#f97316', bg: '#fffbeb' },
  red:    { text: '#dc2626', bar: '#ef4444', bg: '#fef2f2' },
  green:  { text: '#16a34a', bar: '#22c55e', bg: '#f0fdf4' },
  blue:   { text: '#2563eb', bar: '#3b82f6', bg: '#eff6ff' },
}

function chartColor(tab) {
  return { soil: '#f97316', co2: '#ef4444', temp: '#22c55e', humidity: '#22c55e', light: '#22c55e', water: '#3b82f6' }[tab] ?? '#6b7280'
}

// ── TrainModal ────────────────────────────────────────────────────────────────
function TrainModal({ trainStatus, onClose, onTrainDone }) {
  const [excludeManual, setExcludeManual] = useState(false)
  const [ranges, setRanges]               = useState([])
  const [preview, setPreview]             = useState(null)
  const [isPreviewing, setIsPreviewing]   = useState(false)
  const [isTraining, setIsTraining]       = useState(false)
  const [result, setResult]               = useState(null)

  const H = { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' }

  const addRange    = () => setRanges(r => [...r, { start: '', end: '' }])
  const removeRange = (i) => setRanges(r => r.filter((_, j) => j !== i))
  const updateRange = (i, field, val) =>
    setRanges(r => r.map((item, j) => j === i ? { ...item, [field]: val } : item))

  const validRanges = ranges.filter(r => r.start && r.end)

  // 설정 바뀔 때마다 500ms 후 자동 미리보기
  useEffect(() => {
    const timer = setTimeout(async () => {
      setIsPreviewing(true)
      try {
        const res = await fetch(`${API_BASE_URL}/api/train-preview`, {
          method: 'POST', headers: H,
          body: JSON.stringify({ exclude_manual: excludeManual, exclude_ranges: validRanges }),
        })
        if (!res.ok) return   // 404 / 422 등 HTTP 에러 → 무시
        const d = await res.json()
        if (d.ok !== false && d.total !== undefined) setPreview(d)
      } catch { /* 연결 안 됐을 때 조용히 무시 */ }
      finally { setIsPreviewing(false) }
    }, 500)
    return () => clearTimeout(timer)
  }, [excludeManual, JSON.stringify(validRanges)])

  const handleTrain = async () => {
    setIsTraining(true)
    try {
      const res = await fetch(`${API_BASE_URL}/api/train`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ exclude_manual: excludeManual, exclude_ranges: validRanges }),
      })
      const d = await res.json()
      setResult(d)
      onTrainDone()
    } catch (e) { console.error(e) }
    finally { setIsTraining(false) }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 100, display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}>
      <div style={{ background: '#fff', borderRadius: '20px 20px 0 0', width: '100%', maxWidth: 480, padding: '24px 20px 36px', maxHeight: '90vh', overflowY: 'auto' }}>

        {/* 헤더 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div style={{ fontWeight: 700, fontSize: 17 }}>🧠 모델 재학습 설정</div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: '#9ca3af' }}>✕</button>
        </div>

        {/* 완료 상태 */}
        {result && (
          <div style={{ background: '#f0fdf4', borderRadius: 12, padding: '14px 16px', marginBottom: 16, border: '1px solid #bbf7d0' }}>
            <div style={{ fontWeight: 600, color: '#16a34a', marginBottom: 4 }}>✅ 학습 완료!</div>
            <div style={{ color: '#374151', fontSize: 13 }}>
              학습 데이터 {preview?.will_train ?? '-'}건으로 재학습되었습니다.
            </div>
            <button onClick={onClose} style={{ marginTop: 12, width: '100%', background: '#16a34a', color: '#fff', border: 'none', borderRadius: 10, padding: '10px', fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit' }}>닫기</button>
          </div>
        )}

        {!result && (<>
          {/* 현재 데이터 현황 */}
          <div style={{ background: '#f9fafb', borderRadius: 12, padding: '12px 14px', marginBottom: 16 }}>
            <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 6 }}>현재 수집 데이터</div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div><span style={{ fontWeight: 700, fontSize: 18 }}>{trainStatus.total}</span><span style={{ color: '#6b7280', fontSize: 12, marginLeft: 3 }}>전체</span></div>
              <div><span style={{ fontWeight: 700, fontSize: 18, color: '#2563eb' }}>{trainStatus.manual}</span><span style={{ color: '#6b7280', fontSize: 12, marginLeft: 3 }}>수동 제어</span></div>
              <div><span style={{ fontWeight: 700, fontSize: 18, color: '#9ca3af' }}>{trainStatus.total - trainStatus.manual}</span><span style={{ color: '#6b7280', fontSize: 12, marginLeft: 3 }}>자동</span></div>
            </div>
          </div>

          {/* 수동 제어 제외 */}
          <div
            onClick={() => { setExcludeManual(v => !v); setPreview(null) }}
            style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 16px', background: excludeManual ? '#eff6ff' : '#f9fafb', borderRadius: 12, cursor: 'pointer', marginBottom: 12, border: `1px solid ${excludeManual ? '#bfdbfe' : '#e5e7eb'}`, transition: 'all 0.15s' }}
          >
            <div style={{ width: 22, height: 22, borderRadius: 6, border: `2px solid ${excludeManual ? '#2563eb' : '#d1d5db'}`, background: excludeManual ? '#2563eb' : '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'all 0.15s' }}>
              {excludeManual && <span style={{ color: '#fff', fontSize: 13, lineHeight: 1 }}>✓</span>}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14, color: excludeManual ? '#1d4ed8' : '#1c1c1e' }}>수동 제어 구간 제외</div>
              <div style={{ color: '#6b7280', fontSize: 12, marginTop: 1 }}>사용자가 직접 조작한 {trainStatus.manual}건을 학습에서 뺍니다</div>
            </div>
          </div>

          {/* 시간 범위 제외 */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>시간 범위 제외</div>
            {ranges.map((r, i) => (
              <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                <input
                  type="datetime-local" value={r.start}
                  onChange={e => { updateRange(i, 'start', e.target.value); setPreview(null) }}
                  style={{ flex: 1, padding: '8px 10px', border: '1px solid #e5e7eb', borderRadius: 8, fontSize: 13, fontFamily: 'inherit' }}
                />
                <span style={{ color: '#9ca3af', fontSize: 13 }}>~</span>
                <input
                  type="datetime-local" value={r.end}
                  onChange={e => { updateRange(i, 'end', e.target.value); setPreview(null) }}
                  style={{ flex: 1, padding: '8px 10px', border: '1px solid #e5e7eb', borderRadius: 8, fontSize: 13, fontFamily: 'inherit' }}
                />
                <button onClick={() => { removeRange(i); setPreview(null) }} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 18, padding: '0 4px' }}>✕</button>
              </div>
            ))}
            <button
              onClick={addRange}
              style={{ width: '100%', background: 'none', border: '1px dashed #d1d5db', borderRadius: 10, padding: '9px', fontSize: 13, color: '#6b7280', cursor: 'pointer', fontFamily: 'inherit' }}
            >
              + 구간 추가
            </button>
          </div>

          {/* 학습 데이터 미리보기 — 항상 표시, 설정 바뀌면 자동 갱신 */}
          <div style={{ background: preview && preview.will_train < 10 ? '#fef2f2' : '#f0fdf4', borderRadius: 12, padding: '12px 14px', marginBottom: 12, border: `1px solid ${preview && preview.will_train < 10 ? '#fecaca' : '#bbf7d0'}`, transition: 'all 0.2s' }}>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
              학습 데이터 미리보기
              {isPreviewing && <span style={{ color: '#9ca3af' }}>계산 중…</span>}
            </div>
            {preview ? (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 14 }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: 18, color: '#374151' }}>{preview.total}</div>
                    <div style={{ color: '#9ca3af', fontSize: 11 }}>전체</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: 18, color: '#ef4444' }}>−{preview.excluded}</div>
                    <div style={{ color: '#9ca3af', fontSize: 11 }}>제외</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: 18, color: '#16a34a' }}>{preview.will_train}</div>
                    <div style={{ color: '#9ca3af', fontSize: 11 }}>학습 예정</div>
                  </div>
                </div>
                {/* 제외 항목 상세 */}
                {preview.excluded > 0 && (
                  <div style={{ marginTop: 8, fontSize: 12, color: '#6b7280', borderTop: '1px solid #e5e7eb', paddingTop: 8 }}>
                    {excludeManual && <div>· 수동 제어 데이터 {trainStatus.manual}건 제외</div>}
                    {validRanges.map((r, i) => (
                      <div key={i}>· {r.start.replace('T', ' ')} ~ {r.end.replace('T', ' ')} 구간 제외</div>
                    ))}
                  </div>
                )}
                {preview.will_train < 10 && (
                  <div style={{ color: '#dc2626', fontSize: 12, marginTop: 6 }}>⚠ 데이터가 너무 적습니다 (최소 10건 필요)</div>
                )}
              </>
            ) : (
              <div style={{ color: '#9ca3af', fontSize: 13 }}>서버 연결 후 자동으로 표시됩니다</div>
            )}
          </div>

          {/* 버튼 */}
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={handleTrain}
              disabled={isTraining || isPreviewing || (preview && preview.will_train < 10)}
              style={{ flex: 1, background: isTraining ? '#93c5fd' : '#2563eb', color: '#fff', border: 'none', borderRadius: 12, padding: '12px', fontSize: 14, fontWeight: 600, cursor: isTraining ? 'default' : 'pointer', fontFamily: 'inherit' }}
            >
              {isTraining ? '학습 중…' : '재학습 시작'}
            </button>
          </div>
        </>)}
      </div>
    </div>
  )
}

// ── ModelThresholdPanel ───────────────────────────────────────────────────────
const DEVICE_KO = { fan1: '팬 1', fan2: '팬 2', window1: '창문 1', window2: '창문 2', heater: '히터', humidifier: '가습기', led: 'LED' }
const FEATURE_KO = { temp1: '온도', hum1: '습도', light_raw: '조도' }

function ModelThresholdPanel({ currentVersion }) {
  const [data, setData]       = useState(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/model-thresholds`, { headers: { 'ngrok-skip-browser-warning': '1' } })
      .then(r => r.json())
      .then(d => d.ok !== false && setData(d.thresholds))
      .catch(() => {})
  }, [currentVersion])   // 재학습·롤백 시 갱신

  if (!data || Object.keys(data).length === 0) return null

  // fan1/fan2, window1/window2 중복 제거 → 대표 1개씩
  const SHOW = ['fan1', 'window1', 'heater', 'humidifier', 'led']
  const rows = SHOW.map(t => ({ target: t, ...data[t] })).filter(r => r.learned != null)

  return (
    <div style={{ background: '#f9fafb', borderRadius: 12, padding: '12px 14px', marginTop: 10, border: '1px solid #e5e7eb' }}>
      <div
        onClick={() => setExpanded(v => !v)}
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>📊 AI가 학습한 기준값</div>
        <span style={{ color: '#9ca3af', fontSize: 12 }}>{expanded ? '▲ 접기' : '▼ 펼치기'}</span>
      </div>

      {expanded && (
        <div style={{ marginTop: 12 }}>
          {/* 헤더 */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 4, paddingBottom: 6, borderBottom: '1px solid #e5e7eb', marginBottom: 6 }}>
            {['기기', '기준 센서', '규칙 기반', 'AI 학습값'].map(h => (
              <span key={h} style={{ fontSize: 11, color: '#9ca3af', fontWeight: 500 }}>{h}</span>
            ))}
          </div>
          {rows.map(row => {
            const diff = row.diff ?? 0
            const diffColor = diff === 0 ? '#9ca3af' : diff > 0 ? '#2563eb' : '#dc2626'
            const diffStr   = diff === 0 ? '동일' : `${diff > 0 ? '+' : ''}${diff}${row.unit}`
            return (
              <div key={row.target} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 4, padding: '7px 0', borderBottom: '1px solid #f3f4f6', alignItems: 'center' }}>
                <span style={{ fontSize: 13, fontWeight: 500, color: '#1c1c1e' }}>{DEVICE_KO[row.target]}</span>
                <span style={{ fontSize: 12, color: '#6b7280' }}>{FEATURE_KO[row.feature] ?? row.feature}</span>
                <span style={{ fontSize: 12, color: '#374151' }}>{row.rule}{row.unit} {row.dir}</span>
                <div>
                  <span style={{ fontSize: 12, fontWeight: 600, color: '#1c1c1e' }}>{row.learned}{row.unit} {row.dir}</span>
                  <span style={{ fontSize: 11, color: diffColor, marginLeft: 4 }}>({diffStr})</span>
                </div>
              </div>
            )
          })}
          <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 8 }}>
            * AI 학습값은 Random Forest 트리 분기 임계값의 중앙값
          </div>
        </div>
      )}
    </div>
  )
}

// ── ModelVersionPanel ─────────────────────────────────────────────────────────
function ModelVersionPanel({ currentVersion, onRollback }) {
  const [versions, setVersions]   = useState([])
  const [expanded, setExpanded]   = useState(false)
  const [rolling, setRolling]     = useState(null)   // 롤백 중인 version string

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/model-versions`, { headers: { 'ngrok-skip-browser-warning': '1' } })
      .then(r => r.json())
      .then(d => setVersions(d.versions ?? []))
      .catch(() => {})
  }, [currentVersion])

  const handleRollback = async (version) => {
    if (!window.confirm(`${version}으로 롤백할까요?`)) return
    setRolling(version)
    try {
      await fetch(`${API_BASE_URL}/api/model-rollback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
        body: JSON.stringify({ version }),
      })
      onRollback(version)
      setVersions(v => v)   // 재렌더
    } catch (e) { console.error(e) }
    finally { setRolling(null) }
  }

  const fmtDate = (s) => s ? new Date(s).toLocaleString('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '-'

  // train_acc = {target: float} 구버전 또는 {target: {train, test}} 신버전 모두 지원
  const avgAcc = (acc) => {
    const vals = Object.values(acc ?? {})
    if (!vals.length) return null
    const toNum = (v) => typeof v === 'object' ? (v.test ?? v.train ?? 0) : v
    const avg = vals.reduce((s, v) => s + toNum(v), 0) / vals.length
    return Math.round(avg * 100)
  }
  const avgTrainAcc = (acc) => {
    const vals = Object.values(acc ?? {})
    if (!vals.length) return null
    const toNum = (v) => typeof v === 'object' ? (v.train ?? 0) : v
    return Math.round(vals.reduce((s, v) => s + toNum(v), 0) / vals.length * 100)
  }

  return (
    <div style={{ background: '#f9fafb', borderRadius: 12, padding: '12px 14px', marginTop: 10, border: '1px solid #e5e7eb' }}>
      <div
        onClick={() => versions.length > 0 && setExpanded(v => !v)}
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: versions.length > 0 ? 'pointer' : 'default' }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
          📦 모델 버전 기록 ({versions.length}개)
        </div>
        {versions.length > 0
          ? <span style={{ color: '#9ca3af', fontSize: 12 }}>{expanded ? '▲ 접기' : '▼ 펼치기'}</span>
          : <span style={{ color: '#d1d5db', fontSize: 12 }}>재학습 후 생성됩니다</span>
        }
      </div>

      {expanded && versions.length > 0 && (
        <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {versions.map((v) => {
            const isCurrent = v.version === currentVersion
            const acc = avgAcc(v.train_acc)
            return (
              <div
                key={v.version}
                style={{ background: isCurrent ? '#eff6ff' : '#fff', borderRadius: 10, padding: '10px 12px', border: `1px solid ${isCurrent ? '#bfdbfe' : '#e5e7eb'}` }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <span style={{ fontWeight: 700, fontSize: 14, color: isCurrent ? '#1d4ed8' : '#1c1c1e' }}>
                      {v.version}
                    </span>
                    {isCurrent && (
                      <span style={{ marginLeft: 6, fontSize: 11, background: '#2563eb', color: '#fff', borderRadius: 99, padding: '1px 7px' }}>현재</span>
                    )}
                  </div>
                  {!isCurrent && (
                    <button
                      onClick={() => handleRollback(v.version)}
                      disabled={rolling === v.version}
                      style={{ background: '#fee2e2', color: '#dc2626', border: 'none', borderRadius: 8, padding: '4px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit' }}
                    >
                      {rolling === v.version ? '…' : '롤백'}
                    </button>
                  )}
                </div>
                <div style={{ color: '#6b7280', fontSize: 12, marginTop: 4 }}>
                  {fmtDate(v.trained_at)} · 학습 {v.train_count || v.data_count}건 / 검증 {v.val_count || '-'}건 / 테스트 {v.test_count || '-'}건
                </div>
                {avgAcc(v.train_acc) !== null && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    <span style={{ fontSize: 11, background: '#dbeafe', color: '#1d4ed8', borderRadius: 99, padding: '1px 8px' }}>
                      학습 {avgTrainAcc(v.train_acc)}%
                    </span>
                    <span style={{ fontSize: 11, background: avgAcc(v.train_acc) >= 80 ? '#dcfce7' : '#fef3c7', color: avgAcc(v.train_acc) >= 80 ? '#16a34a' : '#d97706', borderRadius: 99, padding: '1px 8px' }}>
                      검증 {avgAcc(v.train_acc)}%
                    </span>
                    {v.exclude_manual && <span style={{ fontSize: 11, background: '#f3f4f6', color: '#6b7280', borderRadius: 99, padding: '1px 8px' }}>수동제외</span>}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Home page ─────────────────────────────────────────────────────────────────
function HomePage({ data, onGoSensor }) {
  const [controlMode, setControlMode] = useState('auto') // 'auto' | 'ai' | 'manual'
  const [watered, setWatered]         = useState(false)
  const [expanded, setExpanded]       = useState(null)
  const [trainStatus, setTrainStatus] = useState({ total: 0, manual: 0, lastTrained: null, modelExists: false })
  const [showTrainModal, setShowTrainModal] = useState(false)
  const [currentVersion, setCurrentVersion] = useState(null)

  // ── 타이머 ────────────────────────────────────────────────────────────────
  const [activeTimers,  setActiveTimers]  = useState([])   // 서버 원본 (5초 동기화)
  const [displayTimers, setDisplayTimers] = useState([])   // 1초 카운트다운용 로컬 복사
  const [timerPanels,   setTimerPanels]   = useState({})   // device → { mode, duration, onTime, offTime }

  const H = { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' }

  const refreshTimers = () => {
    fetch(`${API_BASE_URL}/api/timers`, { headers: { 'ngrok-skip-browser-warning': '1' } })
      .then(r => r.json())
      .then(d => setActiveTimers(d.timers ?? []))
      .catch(() => {})
  }

  // 서버 5초 동기화
  useEffect(() => {
    refreshTimers()
    const id = setInterval(refreshTimers, 5000)
    return () => clearInterval(id)
  }, [])

  // 서버 데이터가 바뀌면 displayTimers 즉시 반영
  useEffect(() => { setDisplayTimers(activeTimers) }, [activeTimers])

  // 1초마다 ends_at / off_at 기준으로 남은 시간 재계산
  useEffect(() => {
    const id = setInterval(() => {
      setDisplayTimers(prev => prev.map(t => {
        const now = Date.now()
        if (t.type === 'duration' && t.ends_at && t.status === 'running') {
          return { ...t, remaining_seconds: Math.max(0, Math.floor((new Date(t.ends_at) - now) / 1000)) }
        }
        if (t.type === 'schedule' && t.status === 'on' && t.off_at) {
          return { ...t, remaining_seconds: Math.max(0, Math.floor((new Date(t.off_at) - now) / 1000)) }
        }
        if (t.type === 'schedule' && t.status === 'waiting' && t.on_at) {
          return { ...t, remaining_until_on: Math.max(0, Math.floor((new Date(t.on_at) - now) / 1000)) }
        }
        return t
      }))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const toggleTimerPanel = (device) => {
    setTimerPanels(prev => ({
      ...prev,
      [device]: prev[device] ? null : { mode: 'duration', duration: '10', onTime: '', offTime: '' },
    }))
  }

  const sendTimer = async (device) => {
    const panel = timerPanels[device]
    if (!panel) return
    const body = panel.mode === 'duration'
      ? { device, timer_type: 'duration', duration_minutes: parseInt(panel.duration) || 10 }
      : { device, timer_type: 'schedule', on_time: panel.onTime, off_time: panel.offTime }
    try {
      await fetch(`${API_BASE_URL}/api/timer`, { method: 'POST', headers: H, body: JSON.stringify(body) })
      setTimerPanels(prev => ({ ...prev, [device]: null }))
      refreshTimers()
    } catch (e) { console.error(e) }
  }

  const cancelTimer = async (timerId) => {
    try {
      await fetch(`${API_BASE_URL}/api/timer/${timerId}`, { method: 'DELETE', headers: { 'ngrok-skip-browser-warning': '1' } })
      refreshTimers()
    } catch (e) { console.error(e) }
  }

  // 기기 상태: 백엔드 데이터로 초기화, 수동 모드에서 사용자가 변경 가능
  const [devOnState, setDevOnState] = useState(() => data?.deviceStates ?? [true, true, false, false, true, false, false])
  useEffect(() => {
    if (data?.deviceStates) setDevOnState(data.deviceStates)
  }, [data])

  // 학습 현황 조회
  const refreshTrainStatus = () => {
    fetch(`${API_BASE_URL}/api/train-status`, { headers: { 'ngrok-skip-browser-warning': '1' } })
      .then(r => r.json())
      .then(d => setTrainStatus({ total: d.total ?? 0, manual: d.manual ?? 0, lastTrained: d.last_trained ?? null, modelExists: d.model_exists ?? false }))
      .catch(() => {})
    fetch(`${API_BASE_URL}/api/model-versions`, { headers: { 'ngrok-skip-browser-warning': '1' } })
      .then(r => r.json())
      .then(d => setCurrentVersion(d.current ?? null))
      .catch(() => {})
  }

  useEffect(() => { refreshTrainStatus() }, [])

  const changeMode = (mode) => {
    setControlMode(mode)
    fetch(`${API_BASE_URL}/api/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify({ mode }),
    }).catch(console.error)
  }

  const toggleDevice = (idx, val) => {
    // 수동 모드가 아니면 자동으로 수동 모드 전환
    if (controlMode !== 'manual') {
      changeMode('manual')
    }
    setDevOnState(prev => prev.map((v, i) => i === idx ? val : v))
    fetch(`${API_BASE_URL}/api/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify({ device: DEVICE_META[idx].name, state: val }),
    }).catch(console.error)
  }

  const autoMode   = controlMode === 'auto'
  const aiMode     = controlMode === 'ai'
  const manualMode = controlMode === 'manual'

  const { statusBars, plantSummary } = data ?? {
    statusBars:   [],
    plantSummary: { title: '불러오는 중…', subtitle: '' },
  }

  const needsWater = data?.latest && parseInt((data.latest.sensors ?? data.latest).soil_percent) < 40
  const needsVent  = data?.latest && parseInt((data.latest.sensors ?? data.latest).co2_raw) > 1000
  const fan1On     = devOnState[0]
  const fan2On     = devOnState[1]

  return (
    <div style={{ maxWidth: 480, margin: '0 auto', padding: '0 0 40px' }}>

      {/* TrainModal */}
      {showTrainModal && (
        <TrainModal
          trainStatus={trainStatus}
          onClose={() => setShowTrainModal(false)}
          onTrainDone={() => { refreshTrainStatus(); setShowTrainModal(false) }}
        />
      )}

      {/* ── Plant Header ── */}
      <div style={{ padding: '20px 20px 0', display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ width: 48, height: 48, borderRadius: '50%', background: '#dcfce7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>🌿</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 17 }}>홍콩야자</div>
          <div style={{ color: '#6b7280', fontSize: 13 }}>거실 창가 · 37일째</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#22c55e' }} />
          <span style={{ color: '#22c55e', fontSize: 12, fontWeight: 600 }}>실시간</span>
        </div>
      </div>

      {/* ── Plant Status Card ── */}
      <div style={{ margin: '16px 16px 0', background: '#fff', borderRadius: 16, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
          <div style={{ width: 40, height: 40, borderRadius: '50%', background: '#dcfce7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>🌿</div>
          <div>
            <div style={{ color: '#6b7280', fontSize: 12, marginBottom: 2 }}>오늘 식물 상태</div>
            <div style={{ fontWeight: 700, fontSize: 18 }}>{plantSummary.title}</div>
            <div style={{ color: '#6b7280', fontSize: 13 }}>{plantSummary.subtitle}</div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {statusBars.map(({ key, icon, label, fill, color, text, detail }) => {
            const isOpen = expanded === key
            return (
              <div key={key}>
                <div
                  onClick={() => setExpanded(isOpen ? null : key)}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 0', cursor: 'pointer' }}
                >
                  <span style={{ fontSize: 16, width: 22, textAlign: 'center' }}>{icon}</span>
                  <span style={{ fontSize: 13, color: '#374151', width: 72, flexShrink: 0 }}>{label}</span>
                  <div style={{ flex: 1, height: 6, background: '#f3f4f6', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{ width: `${fill}%`, height: '100%', background: COLOR[color].bar, borderRadius: 99, transition: 'width 0.6s ease' }} />
                  </div>
                  <span style={{ fontSize: 12, color: COLOR[color].text, width: 64, textAlign: 'right', fontWeight: 500 }}>{text}</span>
                </div>
                {isOpen && (
                  <div style={{ margin: '0 0 8px 32px', padding: '10px 12px', background: COLOR[color].bg, borderRadius: 10, fontSize: 13, color: COLOR[color].text, lineHeight: 1.5 }}>
                    {detail}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        <div style={{ marginTop: 8, color: '#9ca3af', fontSize: 12, textAlign: 'center' }}>항목을 누르면 이유를 알 수 있어요</div>
      </div>

      {/* ── 지금 해주세요 ── */}
      <div style={{ padding: '20px 16px 0' }}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 10 }}>지금 해주세요</div>

        {needsWater && (
          <div style={{ background: '#fffbeb', borderRadius: 14, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
            <span style={{ fontSize: 22 }}>🤚</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>직접 물을 줘야 해요</div>
              <div style={{ color: '#78716c', fontSize: 12, marginTop: 2 }}>토양 수분이 부족해요 · 물을 주세요</div>
            </div>
            <button
              onClick={() => setWatered(v => !v)}
              style={{ background: watered ? '#22c55e' : '#1c1c1e', color: '#fff', border: 'none', borderRadius: 20, padding: '6px 14px', fontSize: 13, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap', fontFamily: 'inherit' }}
            >
              {watered ? '✓ 줬어요' : '줬어요 ✓'}
            </button>
          </div>
        )}

        {(needsVent || fan1On || fan2On) && (
          <div style={{ background: '#eff6ff', borderRadius: 14, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 22 }}>☴</span>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1d4ed8' }}>AI가 환기를 켰어요</div>
              <div style={{ color: '#6b7280', fontSize: 12, marginTop: 2 }}>창문도 열어주시면 공기가 더 빨리 맑아져요</div>
            </div>
          </div>
        )}

        {!needsWater && !needsVent && !fan1On && !fan2On && (
          <div style={{ background: '#f0fdf4', borderRadius: 14, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 22 }}>✅</span>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#16a34a' }}>AI가 알아서 관리 중이에요</div>
              <div style={{ color: '#6b7280', fontSize: 12, marginTop: 2 }}>지금은 특별히 해야 할 일이 없어요</div>
            </div>
          </div>
        )}
      </div>

      {/* ── AI / 수동 제어 섹션 ── */}
      <div style={{ padding: '20px 16px 0' }}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 10 }}>
          {controlMode === 'auto' ? '자동으로 제어 중이에요' : controlMode === 'ai' ? 'AI 모델이 제어 중이에요' : '수동으로 제어 중이에요'}
        </div>

        {/* AI 모드 카드 */}
        {controlMode === 'ai' && (
          <div style={{ background: '#eff6ff', borderRadius: 14, padding: '14px 16px', marginBottom: 10, border: '1px solid #bfdbfe' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1d4ed8' }}>🧠 학습 모델 제어 중</div>
                <div style={{ color: '#3b82f6', fontSize: 12, marginTop: 2 }}>
                  수집 데이터 {trainStatus.total}개 (수동 {trainStatus.manual}개)
                  {currentVersion && <span style={{ marginLeft: 6, background: '#2563eb', color: '#fff', borderRadius: 99, padding: '1px 7px', fontSize: 11 }}>{currentVersion}</span>}
                </div>
                {trainStatus.lastTrained && (
                  <div style={{ color: '#6b7280', fontSize: 11, marginTop: 2 }}>
                    마지막 학습: {new Date(trainStatus.lastTrained).toLocaleString('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                  </div>
                )}
                {!trainStatus.modelExists && (
                  <div style={{ color: '#dc2626', fontSize: 11, marginTop: 2 }}>⚠ 학습된 모델 없음 — 재학습 필요</div>
                )}
              </div>
              <button
                onClick={() => setShowTrainModal(true)}
                style={{ background: '#2563eb', color: '#fff', border: 'none', borderRadius: 10, padding: '8px 14px', fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', whiteSpace: 'nowrap' }}
              >
                재학습
              </button>
            </div>
            <ModelThresholdPanel currentVersion={currentVersion} />
            <ModelVersionPanel
              currentVersion={currentVersion}
              onRollback={(v) => { setCurrentVersion(v); refreshTrainStatus() }}
            />
          </div>
        )}

        <div style={{ background: '#fff', borderRadius: 16, overflow: 'hidden', boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>

          {/* 모드 선택 */}
          <div style={{ display: 'flex', alignItems: 'center', padding: '14px 16px', borderBottom: '1px solid #f3f4f6' }}>
            <span style={{ fontSize: 18, marginRight: 10 }}>
              {controlMode === 'auto' ? '🤖' : controlMode === 'ai' ? '🧠' : '✋'}
            </span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>
                {controlMode === 'auto' ? '자동 모드' : controlMode === 'ai' ? 'AI 모드' : '수동 모드'}
              </div>
              <div style={{ color: '#9ca3af', fontSize: 12 }}>
                {controlMode === 'auto' ? '규칙 기반으로 센서를 보고 알아서 제어해요' : controlMode === 'ai' ? '학습된 모델이 최적 제어를 판단해요' : '기기를 직접 켜고 끌 수 있어요. 자동·AI 전환 시 해제돼요'}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {[['auto', '자동'], ['ai', 'AI'], ['manual', '수동']].map(([mode, label]) => (
                <button
                  key={mode}
                  onClick={() => changeMode(mode)}
                  style={{
                    padding: '5px 14px', borderRadius: 20, border: 'none', fontSize: 12, fontWeight: 600,
                    cursor: 'pointer', fontFamily: 'inherit',
                    background: controlMode === mode ? (mode === 'ai' ? '#2563eb' : mode === 'manual' ? '#d97706' : '#22c55e') : '#f3f4f6',
                    color: controlMode === mode ? '#fff' : '#6b7280',
                    transition: 'background 0.15s',
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* 기기 목록 */}
          {DEVICE_META.map((meta, i) => {
            const isOn       = devOnState[i] ?? false
            const onLbl      = meta.type === 'open-close' ? '열기' : '켜기'
            const offLbl     = meta.type === 'open-close' ? '닫기' : '끄기'
            const statusLbl  = meta.type === 'open-close' ? (isOn ? '열림' : '닫힘') : (isOn ? '켜짐' : '대기')
            const badgeBg    = (statusLbl === '켜짐' || statusLbl === '열림') ? '#dcfce7' : '#f3f4f6'
            const badgeText  = (statusLbl === '켜짐' || statusLbl === '열림') ? '#16a34a' : '#9ca3af'
            const autoDesc   = isOn ? meta.autoDescOn : meta.autoDescOff
            const manualDescOn  = meta.type === 'open-close' ? '사용자가 열었어요' : '사용자가 켰어요'
            const manualDescOff = meta.type === 'open-close' ? '사용자가 닫았어요' : '사용자가 껐어요'
            const manualDesc = isOn ? manualDescOn : manualDescOff

            const devTimers  = displayTimers.filter(t => t.device === meta.name)
            const panel      = timerPanels[meta.name]
            const panelOpen  = !!panel

            const fmtSec = (sec) => {
              if (sec == null) return ''
              const h = Math.floor(sec / 3600)
              const m = Math.floor((sec % 3600) / 60)
              const s = sec % 60
              if (h > 0) return `${h}시간 ${m}분 후`
              if (m > 0) return `${m}분 ${String(s).padStart(2,'0')}초 후`
              return `${s}초 후`
            }

            // duration 타이머 진행률 (0~100)
            const durationProgress = (t) => {
              if (t.type !== 'duration' || !t.duration_minutes) return 0
              const total = t.duration_minutes * 60
              return Math.max(0, Math.min(100, Math.round((1 - t.remaining_seconds / total) * 100)))
            }

            const canSubmit = panel && (
              panel.mode === 'duration'
                ? parseInt(panel.duration) > 0
                : panel.onTime && panel.offTime
            )

            return (
              <div key={i} style={{ borderBottom: i < DEVICE_META.length - 1 ? '1px solid #f3f4f6' : 'none' }}>

                {/* ── 메인 행 ── */}
                <div style={{ display: 'flex', alignItems: 'center', padding: '13px 16px' }}>
                  <span style={{ fontSize: 18, marginRight: 10, width: 24, textAlign: 'center' }}>{meta.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 14, fontWeight: 500 }}>
                      {meta.name}
                      {meta.sub && <span style={{ color: '#9ca3af', fontWeight: 400, fontSize: 12, marginLeft: 4 }}>{meta.sub}</span>}
                    </div>
                    <div style={{ color: '#9ca3af', fontSize: 12, marginTop: 1 }}>{manualMode ? manualDesc : autoDesc}</div>
                  </div>

                  {/* 제어 영역 */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {manualMode ? (
                      <>
                        <button onClick={() => toggleDevice(i, true)}  style={{ padding: '5px 12px', borderRadius: 20, border: 'none', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', background: isOn  ? '#d97706' : '#f3f4f6', color: isOn  ? '#fff' : '#6b7280', transition: 'background 0.15s' }}>{onLbl}</button>
                        <button onClick={() => toggleDevice(i, false)} style={{ padding: '5px 12px', borderRadius: 20, border: 'none', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', background: !isOn ? '#d97706' : '#f3f4f6', color: !isOn ? '#fff' : '#6b7280', transition: 'background 0.15s' }}>{offLbl}</button>
                      </>
                    ) : (
                      <span style={{ fontSize: 12, fontWeight: 600, padding: '3px 10px', borderRadius: 20, background: badgeBg, color: badgeText }}>{statusLbl}</span>
                    )}
                    {/* 타이머 버튼 */}
                    <button
                      onClick={() => toggleTimerPanel(meta.name)}
                      title="타이머 설정"
                      style={{ padding: '5px 9px', borderRadius: 20, border: 'none', fontSize: 13, cursor: 'pointer', fontFamily: 'inherit', background: panelOpen ? '#fef3c7' : '#f3f4f6', color: panelOpen ? '#d97706' : '#9ca3af', transition: 'background 0.15s' }}
                    >⏱</button>
                  </div>
                </div>

                {/* ── 활성 타이머 카드 ── */}
                {devTimers.map(t => {
                  const isDuration = t.type === 'duration'
                  const isWaiting  = t.status === 'waiting'
                  const isOn       = t.status === 'on'
                  const isRunning  = t.status === 'running'

                  let label = '', sublabel = '', progress = null, progressColor = '#d97706'

                  if (isDuration) {
                    if (isRunning) {
                      progress = durationProgress(t)
                      label    = `${fmtSec(t.remaining_seconds)} 꺼짐`
                      sublabel = `${t.duration_minutes}분 타이머 · ${progress}% 경과`
                      progressColor = '#d97706'
                    } else {
                      label = t.status === 'starting' ? '시작 중…' : t.status
                    }
                  } else {
                    // schedule
                    if (isWaiting) {
                      label    = `${t.on_time} 켜짐 예정`
                      sublabel = t.remaining_until_on != null ? `${fmtSec(t.remaining_until_on)} 후 시작` : ''
                      progressColor = '#6b7280'
                    } else if (isOn) {
                      progress = t.remaining_seconds != null && t.on_at && t.off_at
                        ? Math.max(0, Math.min(100, Math.round(
                            (1 - t.remaining_seconds / ((new Date(t.off_at) - new Date(t.on_at)) / 1000)) * 100
                          )))
                        : null
                      label    = `${t.off_time} 꺼짐 예정`
                      sublabel = t.remaining_seconds != null ? `${fmtSec(t.remaining_seconds)} 후 종료` : ''
                      progressColor = '#22c55e'
                    } else {
                      label = t.status
                    }
                  }

                  return (
                    <div key={t.id} style={{ margin: '0 16px 10px 52px', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 10, padding: '10px 12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: progress != null ? 8 : 0 }}>
                        <div>
                          <span style={{ fontSize: 13, fontWeight: 600, color: '#92400e' }}>⏱ {label}</span>
                          {sublabel && <span style={{ fontSize: 11, color: '#b45309', marginLeft: 8 }}>{sublabel}</span>}
                        </div>
                        <button onClick={() => cancelTimer(t.id)} style={{ background: 'none', border: 'none', color: '#d97706', cursor: 'pointer', padding: '0 0 0 8px', fontSize: 14, lineHeight: 1, flexShrink: 0 }}>✕</button>
                      </div>
                      {progress != null && (
                        <div style={{ height: 4, background: '#fef3c7', borderRadius: 99, overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${progress}%`, background: progressColor, borderRadius: 99, transition: 'width 1s linear' }} />
                        </div>
                      )}
                    </div>
                  )
                })}

                {/* ── 타이머 패널 ── */}
                {panel && (
                  <div style={{ margin: '0 16px 12px 52px', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 12, padding: '12px 14px' }}>
                    {/* 모드 탭 */}
                    <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
                      {[['duration', '⏱ 타이머'], ['schedule', '📅 스케줄']].map(([m, lbl]) => (
                        <button
                          key={m}
                          onClick={() => setTimerPanels(prev => ({ ...prev, [meta.name]: { ...prev[meta.name], mode: m } }))}
                          style={{ padding: '5px 12px', borderRadius: 99, border: 'none', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', background: panel.mode === m ? '#d97706' : '#f3f4f6', color: panel.mode === m ? '#fff' : '#6b7280' }}
                        >{lbl}</button>
                      ))}
                    </div>

                    {panel.mode === 'duration' ? (
                      /* 타이머 모드 */
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 13, color: '#78716c', whiteSpace: 'nowrap' }}>켠 후</span>
                        <input
                          type="number" min="1" max="999"
                          value={panel.duration}
                          onChange={e => setTimerPanels(prev => ({ ...prev, [meta.name]: { ...prev[meta.name], duration: e.target.value } }))}
                          style={{ width: 56, padding: '6px 8px', border: '1px solid #fcd34d', borderRadius: 8, fontSize: 14, fontFamily: 'inherit', textAlign: 'center' }}
                        />
                        <span style={{ fontSize: 13, color: '#78716c' }}>분 후 끄기</span>
                      </div>
                    ) : (
                      /* 스케줄 모드 */
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 13, color: '#78716c', whiteSpace: 'nowrap' }}>켜기</span>
                        <input
                          type="time" value={panel.onTime}
                          onChange={e => setTimerPanels(prev => ({ ...prev, [meta.name]: { ...prev[meta.name], onTime: e.target.value } }))}
                          style={{ padding: '6px 8px', border: '1px solid #fcd34d', borderRadius: 8, fontSize: 13, fontFamily: 'inherit' }}
                        />
                        <span style={{ fontSize: 13, color: '#78716c', whiteSpace: 'nowrap' }}>끄기</span>
                        <input
                          type="time" value={panel.offTime}
                          onChange={e => setTimerPanels(prev => ({ ...prev, [meta.name]: { ...prev[meta.name], offTime: e.target.value } }))}
                          style={{ padding: '6px 8px', border: '1px solid #fcd34d', borderRadius: 8, fontSize: 13, fontFamily: 'inherit' }}
                        />
                      </div>
                    )}

                    <button
                      onClick={() => sendTimer(meta.name)}
                      disabled={!canSubmit}
                      style={{ marginTop: 10, width: '100%', background: canSubmit ? '#d97706' : '#e5e7eb', color: canSubmit ? '#fff' : '#9ca3af', border: 'none', borderRadius: 10, padding: '9px', fontSize: 13, fontWeight: 600, cursor: canSubmit ? 'pointer' : 'default', fontFamily: 'inherit' }}
                    >
                      {panel.mode === 'duration' ? `${panel.duration || '-'}분 타이머 시작` : '스케줄 설정'}
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── 센서 페이지 링크 ── */}
      <div style={{ padding: '16px 16px 0' }}>
        <button onClick={onGoSensor} style={{ width: '100%', background: '#fff', border: '1px solid #e5e7eb', borderRadius: 14, padding: '16px 20px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, fontFamily: 'inherit', fontWeight: 600, fontSize: 15, color: '#1c1c1e' }}>
          <span>📊</span><span>센서 수치 · 그래프 · 로그 보기</span><span style={{ color: '#9ca3af' }}>→</span>
        </button>
      </div>
    </div>
  )
}

// ── Sensor page ───────────────────────────────────────────────────────────────
function SensorPage({ data, onBack }) {
  const [activeTab, setActiveTab] = useState('soil')

  const sensors   = data?.sensors   ?? {}
  const chartData = data?.chartData ?? {}
  const logRows   = data?.logRows   ?? []
  const s         = sensors[activeTab] ?? { label: '', range: '', value: '-', unit: '', status: '', color: 'green' }
  const warning   = activeTab === 'soil' && s.status === '부족' ? '부족 — 물주기 필요'
                  : activeTab === 'co2'  && s.status === '높음' ? '높음 — 환기 필요'
                  : null

  return (
    <div style={{ maxWidth: 480, margin: '0 auto', padding: '0 0 40px' }}>

      <div style={{ padding: '20px 16px 0', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 20 }}>센서 수치 · 로그</div>
          <div style={{ color: '#6b7280', fontSize: 13, marginTop: 2 }}>홍콩야자 · 오늘 데이터</div>
        </div>
        <button onClick={onBack} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: 14, fontFamily: 'inherit', padding: '4px 0' }}>← 홈으로</button>
      </div>

      {/* 탭 */}
      <div style={{ padding: '14px 16px 0', display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 2 }}>
        {TABS.map(({ key, label }) => (
          <button key={key} onClick={() => setActiveTab(key)} style={{ flexShrink: 0, padding: '7px 14px', borderRadius: 99, border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 500, background: activeTab === key ? '#1c1c1e' : '#e5e7eb', color: activeTab === key ? '#fff' : '#374151', transition: 'background 0.15s' }}>{label}</button>
        ))}
      </div>

      {/* 차트 카드 */}
      <div style={{ margin: '14px 16px 0', background: '#fff', borderRadius: 16, padding: '18px 18px 14px', boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16 }}>{s.label}</div>
            <div style={{ color: '#9ca3af', fontSize: 12 }}>{s.range}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: 28, fontWeight: 700, color: COLOR[s.color]?.text }}>{s.value}</span>
            <span style={{ fontSize: 13, color: '#9ca3af', marginLeft: 2 }}>{s.unit}</span>
            <div style={{ color: '#9ca3af', fontSize: 11, marginTop: 1 }}>현재 수치</div>
          </div>
        </div>
        {warning && (
          <div style={{ display: 'inline-block', marginBottom: 10, background: COLOR[s.color]?.bg, color: COLOR[s.color]?.text, fontSize: 12, fontWeight: 500, padding: '4px 10px', borderRadius: 8 }}>{warning}</div>
        )}
        <div style={{ height: 120 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData[activeTab] ?? []} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
              <YAxis hide />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }} itemStyle={{ color: chartColor(activeTab) }} formatter={(v) => [`${v}${s.unit}`, s.label]} labelStyle={{ color: '#6b7280' }} />
              <Line type="monotone" dataKey="v" stroke={chartColor(activeTab)} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 현재 센서 수치 2×3 */}
      <div style={{ padding: '14px 16px 0' }}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 10 }}>현재 센서 수치</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {Object.entries(sensors).map(([key, sv]) => (
            <div key={key} style={{ background: '#fff', borderRadius: 14, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
              <div style={{ color: '#6b7280', fontSize: 12, marginBottom: 6 }}>{sv.label}</div>
              <div>
                <span style={{ fontSize: 26, fontWeight: 700, color: COLOR[sv.color]?.text }}>{sv.value}</span>
                <span style={{ fontSize: 13, color: '#9ca3af', marginLeft: 2 }}>{sv.unit}</span>
              </div>
              <div style={{ height: 4, background: '#f3f4f6', borderRadius: 99, margin: '10px 0 8px', overflow: 'hidden' }}>
                <div style={{ height: '100%', borderRadius: 99, background: COLOR[sv.color]?.bar, width: key === 'co2' ? `${Math.min(100, ((sv.value - 400) / 1600) * 100)}%` : key === 'light' ? `${Math.min(100, sv.value / 20)}%` : `${sv.value}%` }} />
              </div>
              <div style={{ color: '#9ca3af', fontSize: 11 }}>{sv.range}</div>
              <div style={{ color: COLOR[sv.color]?.text, fontSize: 13, fontWeight: 600, marginTop: 2 }}>{sv.status}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 센서 기록 테이블 */}
      <div style={{ margin: '14px 16px 0', background: '#fff', borderRadius: 16, padding: 18, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div style={{ fontWeight: 600, fontSize: 15 }}>센서 기록</div>
          <div style={{ color: '#9ca3af', fontSize: 12 }}>오늘 · 2분 간격</div>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ color: '#9ca3af', fontSize: 12 }}>
                {['시간','온도','습도','토양수분','CO₂','조도','수위'].map((h, hi) => (
                  <th key={h} style={{ textAlign: hi === 0 ? 'left' : 'right', paddingBottom: 10, fontWeight: 500, whiteSpace: 'nowrap', paddingRight: hi < 6 ? 12 : 0 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logRows.map((row, i) => (
                <tr key={i} style={{ borderTop: '1px solid #f3f4f6' }}>
                  <td style={{ padding: '10px 12px 10px 0', color: '#6b7280' }}>{row.time}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0' }}>{row.temp}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0' }}>{row.hum}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0', color: row.soilNum < 40 ? '#d97706' : '#1c1c1e', fontWeight: row.soilNum < 40 ? 600 : 400 }}>{row.soil}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0', color: row.co2 > 1000 ? '#dc2626' : '#1c1c1e', fontWeight: row.co2 > 1000 ? 600 : 400 }}>{row.co2}</td>
                  <td style={{ textAlign: 'right', padding: '10px 12px 10px 0' }}>{row.light}</td>
                  <td style={{ textAlign: 'right', padding: '10px 0' }}>{row.water}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage]   = useState('home')
  const [data, setData]   = useState(() => transformRecords(MOCK_RECORDS))
  const [usingMock, setUsingMock] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(API_ENDPOINT, {
        headers: { 'ngrok-skip-browser-warning': '1' },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const records = await res.json()
      const transformed = transformRecords(records)
      if (transformed) {
        setData(transformed)
        setUsingMock(false)
      }
    } catch {
      // 백엔드 미연결 시 mock 데이터 유지
    }
  }, [])

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [fetchData])

  return (
    <div style={{ minHeight: '100vh', background: '#f2f2f7' }}>
      {/* 개발용 mock 표시 배너 (백엔드 연결 후 자동 사라짐) */}
      {usingMock && (
        <div style={{ background: '#fef3c7', borderBottom: '1px solid #fcd34d', padding: '6px 16px', textAlign: 'center', fontSize: 12, color: '#92400e' }}>
          📡 백엔드 미연결 — Mock 데이터 표시 중 ({API_ENDPOINT})
        </div>
      )}
      {page === 'home'
        ? <HomePage   data={data} onGoSensor={() => setPage('sensor')} />
        : <SensorPage data={data} onBack={() => setPage('home')} />
      }
    </div>
  )
}
