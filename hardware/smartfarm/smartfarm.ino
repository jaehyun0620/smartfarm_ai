#include <Servo.h>
#include <DHT.h>

// ===== DHT =====
#define DHTTYPE DHT11
DHT dht1(11, DHTTYPE);
DHT dht2(12, DHTTYPE);

// ===== 릴레이 핀 =====
int fanRelay    = 8;
int fanRelay2   = 9;
int humidRelay  = 10;
int heaterRelay = 3;
int ledRelay    = 4;

// ===== 서보: 창문 2개 =====
Servo servo1;
Servo servo2;

// ===== 마그네틱 센서 핀 =====
int window1MagPin = 2;
int window2MagPin = 7;

// ===== 상태 정의 =====
#define RELAY_ON  HIGH
#define RELAY_OFF LOW
#define HUMID_ON  LOW
#define HUMID_OFF HIGH
#define DEV_ON    LOW
#define DEV_OFF   HIGH

// ===== 센서 핀 =====
int soilPin  = A0;
int lightPin = A1;
int waterPin = A2;
int co2Pin   = A3;

// ===== 창문 각도 =====
const int WINDOW1_OPEN_ANGLE  = 70;
const int WINDOW1_CLOSE_ANGLE = 160;
const int WINDOW2_OPEN_ANGLE  = 100;
const int WINDOW2_CLOSE_ANGLE = 20;

// ===== 기준값 =====
int lightThreshold = 900;
int soilDryValue   = 800;
int soilWetValue   = 350;

// ===== 창문 상태 =====
bool window1Opened = false;
bool window2Opened = false;
bool pumpOn = false;

// ===== 비블로킹 서보 타이머 =====
const unsigned long SERVO_HOLD_MS = 800UL;
bool          servo1Active   = false;
unsigned long servo1StartMs  = 0;
bool          servo2Active   = false;
unsigned long servo2StartMs  = 0;

// ===== 수동 모드 =====
bool manualMode    = false;
bool manualFan1    = false;
bool manualFan2    = false;
bool manualWindow1 = false;
bool manualWindow2 = false;
bool manualLed     = false;
bool manualHumid   = false;
bool manualHeater  = false;

// ===== 타이머 =====
unsigned long lastSendMs   = 0;
unsigned long lastSensorMs = 0;
const unsigned long SEND_INTERVAL   = 10000UL;
const unsigned long SENSOR_INTERVAL = 2000UL;

// ===== 마지막 센서값 =====
float lastTemp1 = 0, lastHum1 = 0, lastTemp2 = 0, lastHum2 = 0;
int   lastSoilRaw = 0, lastSoilPercent = 0;
int   lastLightRaw = 0, lastWaterRaw = 0, lastCo2Raw = 0;
bool  lastFan1On = false, lastFan2On = false;
bool  lastHeaterOn = false, lastHumidOn = false;
bool  lastLedOn = false, lastWindow1Open = false, lastWindow2Open = false;
String lastOverallState = "STABLE", lastEnvironmentState = "STABLE";
String lastTempState = "NORMAL", lastHumState = "NORMAL";
String lastSoilState = "NORMAL", lastLightState = "NORMAL";
bool lastIsDay = true;

// ===================================================
// 구동부 함수
// ===================================================
void allOff() {
  digitalWrite(fanRelay,    RELAY_OFF);
  digitalWrite(fanRelay2,   RELAY_OFF);
  digitalWrite(humidRelay,  HUMID_OFF);
  digitalWrite(heaterRelay, DEV_OFF);
  digitalWrite(ledRelay,    DEV_OFF);
}

void setFan1(bool on) { digitalWrite(fanRelay,  on ? RELAY_ON : RELAY_OFF); }
void setFan2(bool on) { digitalWrite(fanRelay2, on ? RELAY_ON : RELAY_OFF); }
void setFan(bool on)  { setFan1(on); setFan2(on); }

void setHumidifier(bool on) { digitalWrite(humidRelay,  on ? HUMID_ON : HUMID_OFF); }
void setHeater(bool on)     { digitalWrite(heaterRelay, on ? DEV_ON   : DEV_OFF); }
void setLed(bool on)        { digitalWrite(ledRelay,    on ? DEV_ON   : DEV_OFF); }

// delay() 없는 비블로킹 서보 제어
void setWindow1(bool open) {
  if (window1Opened != open) {
    servo1.attach(5);
    servo1.write(open ? WINDOW1_OPEN_ANGLE : WINDOW1_CLOSE_ANGLE);
    servo1Active  = true;
    servo1StartMs = millis();
    window1Opened = open;
  }
}

void setWindow2(bool open) {
  if (window2Opened != open) {
    servo2.attach(6);
    servo2.write(open ? WINDOW2_OPEN_ANGLE : WINDOW2_CLOSE_ANGLE);
    servo2Active  = true;
    servo2StartMs = millis();
    window2Opened = open;
  }
}

void setWindows(bool open) { setWindow1(open); setWindow2(open); }

// loop() 최상단에서 매 턴 호출 — 800ms 후 서보 전원 차단
void tickServos(unsigned long now) {
  if (servo1Active && now - servo1StartMs >= SERVO_HOLD_MS) {
    servo1.detach();
    servo1Active = false;
  }
  if (servo2Active && now - servo2StartMs >= SERVO_HOLD_MS) {
    servo2.detach();
    servo2Active = false;
  }
}

// ===================================================
// 유틸
// ===================================================
int soilToPercent(int raw) {
  return constrain(map(raw, soilDryValue, soilWetValue, 0, 100), 0, 100);
}

bool isWindow1ClosedBySensor() { return digitalRead(window1MagPin) == LOW; }
bool isWindow2ClosedBySensor() { return digitalRead(window2MagPin) == LOW; }

// ===================================================
// 시리얼 명령 수신
// ===================================================
int extractValue(const String& json, const String& key) {
  String search = "\"" + key + "\":";
  int idx = json.indexOf(search);
  if (idx < 0) return -1;
  idx += search.length();
  while (idx < (int)json.length() && json[idx] == ' ') idx++;
  char c = json[idx];
  if (c == '1' || c == 't') return 1;
  return 0;
}

void checkSerialCommand() {
  if (!Serial.available()) return;
  String raw = Serial.readStringUntil('\n');
  raw.trim();
  if (raw.length() == 0) return;

  Serial.print("[CMD]"); Serial.println(raw);

  if (raw.indexOf("\"manual\"") >= 0) {
    manualMode = (extractValue(raw, "manual") == 1);
    return;
  }

  // 수동 모드 최초 진입 시 현재 상태 상속
  if (!manualMode) {
    manualFan1    = lastFan1On;
    manualFan2    = lastFan2On;
    manualWindow1 = lastWindow1Open;
    manualWindow2 = lastWindow2Open;
    manualLed     = lastLedOn;
    manualHumid   = lastHumidOn;
    manualHeater  = lastHeaterOn;
  }

  bool updated = false;
  if (raw.indexOf("\"fan1\"") >= 0)       { manualFan1  = extractValue(raw, "fan1")       == 1; updated = true; }
  if (raw.indexOf("\"fan2\"") >= 0)       { manualFan2  = extractValue(raw, "fan2")       == 1; updated = true; }
  if (raw.indexOf("\"window1\"") >= 0) {
    manualWindow1   = extractValue(raw, "window1") == 1;
    window1Opened   = !manualWindow1;   // 가드 우회: 무조건 이동
    setWindow1(manualWindow1);
    lastWindow1Open = manualWindow1;
    updated = true;
  }
  if (raw.indexOf("\"window2\"") >= 0) {
    manualWindow2   = extractValue(raw, "window2") == 1;
    window2Opened   = !manualWindow2;   // 가드 우회: 무조건 이동
    setWindow2(manualWindow2);
    lastWindow2Open = manualWindow2;
    updated = true;
  }
  if (raw.indexOf("\"led\"") >= 0)        { manualLed   = extractValue(raw, "led")        == 1; updated = true; }
  if (raw.indexOf("\"humidifier\"") >= 0) { manualHumid = extractValue(raw, "humidifier") == 1; updated = true; }
  if (raw.indexOf("\"heater\"") >= 0)     { manualHeater= extractValue(raw, "heater")     == 1; updated = true; }
  if (updated) manualMode = true;
}

// ===================================================
// JSON 출력
// ===================================================
void printJson() {
  bool w1Sensor = isWindow1ClosedBySensor();
  bool w2Sensor = isWindow2ClosedBySensor();

  Serial.print("{");
  Serial.print("\"overall_state\":\"");     Serial.print(lastOverallState);     Serial.print("\",");
  Serial.print("\"environment_state\":\""); Serial.print(lastEnvironmentState); Serial.print("\",");
  Serial.print("\"manual\":"); Serial.print(manualMode ? 1 : 0); Serial.print(",");
  Serial.print("\"states\":{");
  Serial.print("\"temp\":\"");     Serial.print(lastTempState);  Serial.print("\",");
  Serial.print("\"humidity\":\""); Serial.print(lastHumState);   Serial.print("\",");
  Serial.print("\"soil\":\"");     Serial.print(lastSoilState);  Serial.print("\",");
  Serial.print("\"light\":\"");    Serial.print(lastLightState); Serial.print("\",");
  Serial.print("\"co2\":\"MONITOR_ONLY\",");
  Serial.print("\"day_night\":\""); Serial.print(lastIsDay ? "DAY" : "NIGHT"); Serial.print("\"");
  Serial.print("},");
  Serial.print("\"sensors\":{");
  Serial.print("\"temp1\":"); Serial.print(lastTemp1); Serial.print(",");
  Serial.print("\"hum1\":");  Serial.print(lastHum1);  Serial.print(",");
  Serial.print("\"temp2\":"); Serial.print(lastTemp2); Serial.print(",");
  Serial.print("\"hum2\":");  Serial.print(lastHum2);  Serial.print(",");
  Serial.print("\"soil_raw\":"); Serial.print(lastSoilRaw); Serial.print(",");
  Serial.print("\"soil_percent\":"); Serial.print(lastSoilPercent); Serial.print(",");
  Serial.print("\"light_raw\":"); Serial.print(lastLightRaw); Serial.print(",");
  Serial.print("\"water_raw\":"); Serial.print(lastWaterRaw); Serial.print(",");
  Serial.print("\"co2_raw\":"); Serial.print(lastCo2Raw); Serial.print(",");
  Serial.print("\"window1_magnetic\":\""); Serial.print(w1Sensor ? "CLOSED" : "OPEN"); Serial.print("\",");
  Serial.print("\"window2_magnetic\":\""); Serial.print(w2Sensor ? "CLOSED" : "OPEN"); Serial.print("\"");
  Serial.print("},");
  Serial.print("\"actuators\":{");
  Serial.print("\"fan1\":"); Serial.print(lastFan1On ? 1 : 0); Serial.print(",");
  Serial.print("\"fan2\":"); Serial.print(lastFan2On ? 1 : 0); Serial.print(",");
  Serial.print("\"window1\":"); Serial.print(lastWindow1Open ? 1 : 0); Serial.print(",");
  Serial.print("\"window2\":"); Serial.print(lastWindow2Open ? 1 : 0); Serial.print(",");
  Serial.print("\"heater\":"); Serial.print(lastHeaterOn ? 1 : 0); Serial.print(",");
  Serial.print("\"humidifier\":"); Serial.print(lastHumidOn ? 1 : 0); Serial.print(",");
  Serial.print("\"led\":"); Serial.print(lastLedOn ? 1 : 0); Serial.print(",");
  Serial.print("\"pump\":"); Serial.print(pumpOn ? 1 : 0);
  Serial.print("},");
  Serial.print("\"window_status\":{");
  Serial.print("\"window1_command\":\""); Serial.print(lastWindow1Open ? "OPEN" : "CLOSED"); Serial.print("\",");
  Serial.print("\"window2_command\":\""); Serial.print(lastWindow2Open ? "OPEN" : "CLOSED"); Serial.print("\",");
  Serial.print("\"window1_sensor\":\"");  Serial.print(w1Sensor ? "CLOSED" : "OPEN"); Serial.print("\",");
  Serial.print("\"window2_sensor\":\"");  Serial.print(w2Sensor ? "CLOSED" : "OPEN"); Serial.print("\"");
  Serial.print("}");
  Serial.println("}");
}

// ===================================================
// setup
// ===================================================
void setup() {
  Serial.begin(9600);
  Serial.setTimeout(500);

  dht1.begin();
  dht2.begin();

  pinMode(fanRelay,    OUTPUT);
  pinMode(fanRelay2,   OUTPUT);
  pinMode(humidRelay,  OUTPUT);
  pinMode(heaterRelay, OUTPUT);
  pinMode(ledRelay,    OUTPUT);

  pinMode(window1MagPin, INPUT_PULLUP);
  pinMode(window2MagPin, INPUT_PULLUP);

  allOff();

  // setup() 에서만 blocking delay 허용
  servo1.attach(5);
  servo1.write(WINDOW1_CLOSE_ANGLE);
  servo2.attach(6);
  servo2.write(WINDOW2_CLOSE_ANGLE);
  delay(900);
  servo1.detach();
  servo2.detach();

  window1Opened = false;
  window2Opened = false;
}

// ===================================================
// loop
// ===================================================
void loop() {
  unsigned long now = millis();

  tickServos(now);
  checkSerialCommand();

  if (now - lastSensorMs >= SENSOR_INTERVAL) {
    lastSensorMs = now;

    lastTemp1       = dht1.readTemperature();
    lastHum1        = dht1.readHumidity();
    lastTemp2       = dht2.readTemperature();
    lastHum2        = dht2.readHumidity();
    lastSoilRaw     = analogRead(soilPin);
    lastLightRaw    = analogRead(lightPin);
    lastWaterRaw    = analogRead(waterPin);
    lastCo2Raw      = analogRead(co2Pin);
    lastSoilPercent = soilToPercent(lastSoilRaw);

    if (isnan(lastTemp1) || isnan(lastHum1)) {
      Serial.println("{\"overall_state\":\"ERROR\",\"error\":\"DHT1_READ_FAILED\"}");
      return;
    }

    if (manualMode) {
      setFan1(manualFan1);
      setFan2(manualFan2);
      // 창문은 checkSerialCommand() 에서 즉시 처리
      setLed(manualLed);
      setHumidifier(manualHumid);
      setHeater(manualHeater);

      lastFan1On      = manualFan1;
      lastFan2On      = manualFan2;
      lastWindow1Open = manualWindow1;
      lastWindow2Open = manualWindow2;
      lastHeaterOn    = manualHeater;
      lastHumidOn     = manualHumid;
      lastLedOn       = manualLed;
      lastOverallState     = "MANUAL";
      lastEnvironmentState = "MANUAL";
      lastTempState = lastHumState = lastSoilState = lastLightState = "MANUAL";

    } else {
      lastIsDay = true;

      float tempHigh = 22.3;
      float tempLow  = 14.9;
      float humHigh  = 80.6;
      float humLow   = 66.6;

      bool tempTooHigh = lastTemp1 > tempHigh;
      bool tempTooLow  = lastTemp1 < tempLow;
      bool humTooHigh  = lastHum1  > humHigh;
      bool humTooLow   = lastHum1  < humLow;
      bool lightTooLow = lastLightRaw > lightThreshold;

      lastTempState  = "NORMAL";
      lastHumState   = "NORMAL";
      lastSoilState  = "NORMAL";
      lastLightState = "NORMAL";

      if (tempTooHigh)     lastTempState  = "HIGH";
      else if (tempTooLow) lastTempState  = "LOW";
      if (humTooHigh)      lastHumState   = "HIGH";
      else if (humTooLow)  lastHumState   = "LOW";
      if (lightTooLow)     lastLightState = "LOW";

      if (lastSoilPercent < 30)       { pumpOn = true;  lastSoilState = "DRY"; }
      else if (lastSoilPercent >= 70) { pumpOn = false; lastSoilState = "ENOUGH"; }
      else                            {                  lastSoilState = "KEEP"; }

      bool fanOn = false;
      lastHeaterOn = false; lastHumidOn = false;
      lastLedOn = false; lastWindow1Open = false; lastWindow2Open = false;
      lastEnvironmentState = "STABLE";

      if (tempTooHigh) {
        fanOn = true; lastWindow1Open = lastWindow2Open = true;
        lastEnvironmentState = "TEMP_HIGH";
      } else if (tempTooLow) {
        lastHeaterOn = true;
        lastHumidOn  = humTooLow;
        lastEnvironmentState = "TEMP_LOW";
      } else if (humTooHigh) {
        fanOn = true; lastWindow1Open = lastWindow2Open = true;
        lastEnvironmentState = "HUMIDITY_HIGH";
      } else if (humTooLow) {
        lastHumidOn = true;
        lastEnvironmentState = "HUMIDITY_LOW";
      }

      if (lastIsDay && lightTooLow) lastLedOn = true;

      lastFan1On = fanOn;
      lastFan2On = fanOn;

      lastOverallState = (fanOn || lastHeaterOn || lastHumidOn || lastLedOn || lastWindow1Open || pumpOn)
                         ? "ACTIVE" : "STABLE";

      setFan(fanOn);
      setWindow1(lastWindow1Open);
      setWindow2(lastWindow2Open);
      setHeater(lastHeaterOn);
      setHumidifier(lastHumidOn);
      setLed(lastLedOn);
    }
  }

  if (now - lastSendMs >= SEND_INTERVAL) {
    lastSendMs = now;
    printJson();
  }
}
