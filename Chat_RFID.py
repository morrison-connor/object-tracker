import numpy as np
import matplotlib.pyplot as plt
import adi
import time

def generate_waveform(data_bits, symbol_length, high_amp=2**14, low_amp=0):
    waveform = np.array([], dtype=np.float32)
    for bit in data_bits:
        amplitude = high_amp if bit == 1 else low_amp
        symbol = amplitude * np.ones(int(symbol_length))
        waveform = np.append(waveform, symbol)
    return waveform

def miller_encode(bits, miller_type=4):
    # miller encoding (type 2 or 4) which is used to smooth bit transitions in communication
    encoded = []
    state = 1  # EPC Gen2 starts with a high state
    for bit in bits:
        if bit == 0:
            encoded.extend([state, state, 1 - state, 1 - state] if miller_type == 4 else [state, 1 - state])
        else:
            state = 1 - state
            encoded.extend([state, 1 - state, state, 1 - state] if miller_type == 4 else [1 - state, state])
    return np.array(encoded)


def modulate_waveform(waveform, carrier_freq, fs):
    """
    Modulates a baseband waveform using Amplitude Shift Keying (ASK).

    Parameters:
        waveform (np.array): The baseband signal (0s and 1s from Miller encoding).
        carrier_freq (float): Carrier frequency in Hz.
        fs (float): Sampling frequency in Hz.

    Returns:
        np.array: Modulated waveform (real).
        np.array: Time vector for plotting.
    """
    N = len(waveform)
    ts = 1 / float(fs)  # Sampling period
    t = np.arange(0, N * ts, ts)  # Time vector

    # Generate carrier signal (cosine wave for ASK)
    carrier = np.cos(2 * np.pi * carrier_freq * t)

    # Apply modulation (ASK: amplitude changes according to waveform)
    modulated_signal = waveform * carrier

    return modulated_signal, t


def transmit_signal(sdr, signal, carrier_freq, fs):
    sdr.tx_rf_bandwidth = int(fs)
    sdr.tx_lo = int(carrier_freq)
    sdr.tx_hardwaregain_chan0 = -10
    sdr.tx([signal, signal])

def receive_signal(sdr, carrier_freq, fs):
    sdr.rx_enabled_channels = [0]  # TODO only using one channel for now
    sdr.rx_lo = int(carrier_freq)
    sdr.rx_rf_bandwidth = int(fs)
    return sdr.rx()

def plot_signals(tx_t, tx_signal, rx_t, rx_signal, fs):
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(tx_t, tx_signal, color="purple")
    plt.title("Tx Signal - Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    freq = np.fft.fftfreq(len(tx_signal), d=1/fs)
    fft_signal = np.fft.fft(tx_signal)
    plt.subplot(4, 1, 2)
    plt.plot(freq[:int(fs)//2], np.abs(fft_signal[:int(fs)//2]))
    plt.title("Tx Signal - Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(rx_t, rx_signal.real, label="Real", color="blue")
    plt.plot(rx_t, rx_signal.imag, label="Imaginary", color="red", linestyle="--")
    plt.plot(rx_t, np.abs(rx_signal), label="Absolute Value", color="purple", linestyle="-.")
    plt.title("Rx Signal - Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    freq = np.fft.fftfreq(len(rx_signal), d=1/fs)
    fft_signal_rx = np.fft.fft(rx_signal)
    plt.subplot(4, 1, 4)
    plt.plot(freq[:int(fs)//2], np.abs(fft_signal_rx[:int(fs)//2]))
    plt.title("Rx Signal - Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

# Parameters
fs = 1e6  # sample rate
bit_rate = 40e3  # EPS Gen2 standard
symbol_length = fs / bit_rate
carrier_freq = 915e6  # UHF carrier frequency
query_cmd = [1, 0, 1, 1, 0, 0, 1, 0]  # EPC Gen2 query command

# Miller encoding
encoded_bits = miller_encode(query_cmd)

# Generate ASK waveform using encoded bits
waveform = generate_waveform(encoded_bits, symbol_length)

# Modulate waveform onto carrier
# This step is handled by the AD9361 (assuming baseband mode) but is useful to visualize with plotting
modulated_waveform, tx_t = modulate_waveform(waveform, carrier_freq, fs)

# Convert to complex I/Q signal
tx_signal = waveform + 1j * np.zeros_like(waveform)

# SDR setup
sdr = adi.ad9361(uri='ip:192.168.2.1')
sdr.rx_buffer_size = int(symbol_length * len(encoded_bits) * 2)  # adjust dynamically based on cmd length

# Transmit signal
for _ in range(10):  # send out 10 queries
    transmit_signal(sdr, tx_signal, carrier_freq, fs)
    time.sleep(0.01)  # allow time for tag to respond

# Receive signals
rx_signal = receive_signal(sdr, carrier_freq, fs)

# Time vectors for plotting
tx_t = np.linspace(0, (len(tx_signal) - 1) / fs, len(tx_signal))
rx_t = np.linspace(0, (len(rx_signal) - 1) / fs, len(rx_signal))

plot_signals(tx_t, tx_signal, rx_t, rx_signal, fs)
