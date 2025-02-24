import serial
import time
import adi
import numpy as np
import matplotlib.pyplot as plt

def plot_signal(title, signal, sample_rate):
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(12, 6))
    plt.plot(t, np.real(signal), label='I (In-phase)')
    plt.plot(t, np.imag(signal), label='Q (Quadrature)')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

def plot_frequency_domain(signal, sample_rate):
    freq = np.fft.fftfreq(len(signal), 1 / sample_rate)
    fft_signal = np.fft.fft(signal)
    plt.figure(figsize=(12, 6))
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft_signal)))
    plt.title("Received Signal - Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

CENTER_FREQ = int(913e6)
SAMPLE_RATE = 10e6
TX_GAIN = -20       # TX gain (adjust as needed)
RX_GAIN = 0         # RX gain increased to boost weak tag response

sdr = adi.ad9361(uri='ip:192.168.2.1')
sdr.sample_rate = int(SAMPLE_RATE)
sdr.tx_lo = int(CENTER_FREQ)
sdr.tx_hardwaregain_chan0 = TX_GAIN
sdr.rx_lo = int(CENTER_FREQ)
sdr.rx_hardwaregain_chan0 = RX_GAIN
sdr.rx_buffer_size = 32768
sdr.rx_enabled_channels = [0]  # Use only one RX channel

ser = serial.Serial(
    port='COM4',       # Change to the correct COM port
    baudrate=38400,     # Change if necessary
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

command = '\nQ\r'

ser.write(command.encode())
signal = sdr.rx()
plot_signal("TX", signal, SAMPLE_RATE)
plot_frequency_domain(signal, SAMPLE_RATE)

ser.close()
del sdr