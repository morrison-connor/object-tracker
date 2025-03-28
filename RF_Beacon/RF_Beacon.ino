#include <SPI.h>

// --- Pin Assignments ---
const int LE_PIN = 10;     // Latch Enable for ADF4350
const int SWITCH_PIN = 12; // RF switch control pin (OOK modulation)

// --- ADF4350 Register Settings for 2.3 GHz ---
uint32_t adf4350_registers[6] = {
  0x00580008,  // R5
  0x009F0039,  // R4: RF output enable, feedback fundamental, 0dB power
  0x00004E42,  // R3: Band select clock divider = 200
  0x000004B3,  // R2: MUXOUT = digital lock detect, CP current = 2.5mA
  0x08008011,  // R1: MOD = 1, PHASE = 1, Phase adjust = 0
  0x005C8000   // R0: INT = 92, FRAC = 0, MOD = 1
};

void setup() {
  // Initialize control pins
  pinMode(LE_PIN, OUTPUT);
  digitalWrite(LE_PIN, HIGH);

  pinMode(SWITCH_PIN, OUTPUT);
  digitalWrite(SWITCH_PIN, LOW);

  // Initialize SPI
  SPI.begin();  // automatically sets D11=MOSI D13=CLK
  SPI.setDataMode(SPI_MODE0);
  SPI.setClockDivider(SPI_CLOCK_DIV16); // ~1 MHz SPI clock
  SPI.setBitOrder(MSBFIRST);

  delay(10); // Allow power to stabilize

  // Program ADF4350 (R5 → R0)
  for (int i = 5; i >= 0; i--) {
    writeADF4350(adf4350_registers[i]);
    delayMicroseconds(100); // Let the register settle
  }

  pinMode(3, OUTPUT);  // D3 = OC2B

  // CODE FOR 200khz PWM ON PIN 3
  // Stop Timer2
  TCCR2A = 0;
  TCCR2B = 0;
  TCNT2  = 0;

  // Set CTC mode (Clear Timer on Compare Match)
  // Toggle OC2B on Compare Match
  TCCR2A = _BV(COM2B0) | _BV(WGM21);  // Toggle D3 (OC2B), CTC mode
  TCCR2B = _BV(CS20);                 // No prescaler (timer runs at 16 MHz)

  // Set compare value for 200 kHz:
  // 16 MHz / (2 * (OCR2A + 1)) = 200 kHz → OCR2A = 39
  OCR2A = 39;

  // // Code for 20 kHz PWM on PIN 3
  //  // Reset Timer2
  // TCCR2A = 0;
  // TCCR2B = 0;
  // TCNT2  = 0;

  // // CTC mode, toggle OC2B (D3) on compare match
  // TCCR2A = _BV(COM2B0) | _BV(WGM21);  // Toggle OC2B, CTC mode
  // TCCR2B = _BV(CS21);                 // Prescaler = 8

  // OCR2A = 49;  // Results in 20 kHz toggle frequency

  // pinMode(3, OUTPUT);  // D3 = OC2B

  // // Stop Timer2
  // TCCR2A = 0;
  // TCCR2B = 0;
  // TCNT2  = 0;

  // // CTC mode, toggle OC2B on compare match
  // TCCR2A = _BV(COM2B0) | _BV(WGM21);  // Toggle D3, CTC mode
  // TCCR2B = _BV(CS20);                 // Prescaler = 1

  // OCR2A = 9;  // (16 MHz / (2 * 1 * (9 + 1)) = 800 kHz
}

void loop() {
  // // Generate 200 kHz square wave for RF switch control
  // digitalWrite(SWITCH_PIN, HIGH);
  // delayMicroseconds(2);  // High for 2.5 µs
  // digitalWrite(SWITCH_PIN, LOW);
  // delayMicroseconds(2);  // Low for 2.5 µs
}

// --- Function to Write 32-bit Register to ADF4350 ---
void writeADF4350(uint32_t data) {
  digitalWrite(LE_PIN, LOW);
  SPI.transfer((data >> 24) & 0xFF);
  SPI.transfer((data >> 16) & 0xFF);
  SPI.transfer((data >> 8) & 0xFF);
  SPI.transfer(data & 0xFF);
  digitalWrite(LE_PIN, HIGH);
}
