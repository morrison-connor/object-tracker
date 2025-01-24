import numpy as np
import matplotlib.pyplot as plt
import adi

def generate_waveform(query_bits, fs, bit_rate, high_amp=1.0, low_amp=0.0):
    symbol_length = int(fs / bit_rate)
    waveform = np.array([], dtype=np.float32)
    for bit in query_bits:
        amplitude = high_amp if bit == 1 else low_amp
        waveform = np.append(waveform, amplitude * np.ones(symbol_length))
    return waveform

def modulate_waveform(waveform, carrier_freq, fs):
    N = len(waveform)
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    return waveform * carrier, t

def transmit_signal(sdr, signal, carrier_freq, fs):
    sdr.tx_rf_bandwidth = int(fs)
    sdr.tx_lo = int(carrier_freq)
    sdr.tx_hardwaregain_chan0 = -10
    sdr.tx([signal, signal])

def receive_signal(sdr, carrier_freq, fs, buf_len):
    sdr.rx_enabled_channels = [0]
    sdr.rx_lo = int(carrier_freq)
    sdr.rx_rf_bandwidth = int(fs)
    sdr.rx_buffer_size = buf_len
    return sdr.rx(), np.arange(buf_len)

def plot_signals(tx_t, transmit_signal, rx_t, received_signal, fs):
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(tx_t, transmit_signal, label="Transmit Signal", color="purple")
    plt.title("Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    freq = np.fft.fftfreq(len(transmit_signal), d=1/fs)
    fft_signal = np.fft.fft(transmit_signal)
    plt.subplot(4, 1, 2)
    plt.plot(freq[:int(fs)//2], np.abs(fft_signal[:int(fs)//2]))
    plt.title("Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(rx_t, received_signal.real, label="Real", color="blue")
    plt.plot(rx_t, received_signal.imag, label="Imaginary", color="red", linestyle="--")
    plt.plot(np.abs(received_signal), label="Absolute Value", color="purple", linestyle="-.")
    plt.title("Time Domain Received Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    freq = np.fft.fftfreq(len(received_signal), d=1/fs)
    fft_signal_rx = np.fft.fft(received_signal)
    plt.subplot(4, 1, 4)
    plt.plot(freq[:int(fs)//2], np.abs(fft_signal_rx[:int(fs)//2]))
    plt.title("Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

# Parameters
fs = 1e6
bit_rate = 40e3
carrier_freq = 915e6
query_bits = [1, 0, 1, 1, 0, 0, 1, 0]

waveform = generate_waveform(query_bits, fs, bit_rate)
tx_signal, tx_t = modulate_waveform(waveform, carrier_freq, fs)

sdr = adi.ad9361(uri='ip:192.168.2.1')
transmit_signal(sdr, tx_signal, carrier_freq, fs)

rx_signal, rx_t = receive_signal(sdr, carrier_freq, fs, 2**16)
plot_signals(tx_t, tx_signal, rx_t, rx_signal, fs)
