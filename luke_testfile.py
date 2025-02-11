import numpy as np
import adi
import time
import matplotlib.pyplot as plt

def crc5(bits):
    """ Computes CRC-5 checksum for the EPC Gen2 Query command. """
    poly = 0b101001
    crc = 0b11111
    for bit in bits:
        crc_in = (crc >> 4) & 1
        crc = ((crc << 1) & 0b11111) | bit
        if crc_in:
            crc ^= poly
    return [(crc >> i) & 1 for i in reversed(range(5))]

def build_query_command(dr=1, M=2, TRext=1, Sel=0, Session=0, Target=0, Q=4):
    """ Constructs a complete EPC Gen2 Query command with CRC-5. """
    if M not in [1, 2, 4, 8]:
        raise ValueError("M must be 1 (FM0), 2, 4, or 8 (Miller encoding).")
    miller_map = {1: [0, 0], 2: [0, 1], 4: [1, 0], 8: [1, 1]}
    M_bits = miller_map[M]
    query_bits = [1, 0, 0, 0] + [dr] + M_bits + [TRext]
    query_bits += [(Sel >> 1) & 1, Sel & 1]
    query_bits += [(Session >> 1) & 1, Session & 1]
    query_bits += [Target] + [(Q >> i) & 1 for i in range(3, -1, -1)]
    query_bits += crc5(query_bits)
    preamble = [0] * 12 + [1, 0, 1, 0] if TRext == 0 else [1] * 12 + [1, 0, 1, 0]
    return preamble + query_bits

def encode_pie(command_bits, sample_rate=10e6, bitrate=40e3, high=2**14, low=0):
    """
    Encodes an EPC Gen2 command using Pulse Interval Encoding (PIE).

    Args:
        command_bits (list): The binary command to encode.
        sample_rate (float): SDR sample rate in Hz.
        bitrate (float): Bitrate of the tag.
        high (int): High signal level for ASK modulation.
        low (int): Low signal level for ASK modulation.

    Returns:
        np.array: Encoded PIE waveform (ASK-modulated).
    """
    tari = 1 / bitrate  # Base unit time (Tari)
    short_pulse = int(sample_rate * tari)
    long_pulse = int(1.5 * sample_rate * tari)
    waveform = []
    for bit in command_bits:
        if bit == 0:
            waveform.extend([high] * short_pulse)
            waveform.extend([low] * long_pulse)
        else:
            waveform.extend([high] * (2 * short_pulse))
            waveform.extend([low] * short_pulse)
    return np.array(waveform, dtype=np.float32)

def plot_transmitted_waveform(waveform, sample_rate):
    t = np.arange(len(waveform)) / sample_rate
    plt.figure(figsize=(12, 6))
    plt.plot(t, waveform)
    plt.title("Constructed Query Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

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

# --- Main Code Execution ---

# PlutoSDR and System Configuration
CENTER_FREQ = int(915e6)
SAMPLE_RATE = 10e6
TX_GAIN = -20       # TX gain (adjust as needed)
RX_GAIN = 30         # RX gain increased to boost weak tag response

# Build the Query Command and encode it using PIE
dr, m, trext, sel, session, target, q = 1, 2, 0, 0, 0, 0, 4
QUERY_COMMAND = build_query_command(dr, m, trext, sel, session, target, q)
waveform = encode_pie(QUERY_COMMAND, sample_rate=SAMPLE_RATE)

print("Constructed Query Command (bits):", QUERY_COMMAND)
print("Waveform Length (samples):", len(waveform))
print("Waveform Duration (seconds):", len(waveform) / SAMPLE_RATE)
#plot_transmitted_waveform(waveform, SAMPLE_RATE)

# Initialize PlutoSDR and configure TX/RX
sdr = adi.ad9361(uri='ip:192.168.2.1')
sdr.sample_rate = int(SAMPLE_RATE)
sdr.tx_lo = int(CENTER_FREQ)
sdr.tx_hardwaregain_chan0 = TX_GAIN
sdr.rx_lo = int(CENTER_FREQ)
sdr.rx_hardwaregain_chan0 = RX_GAIN
sdr.rx_buffer_size = len(waveform)
sdr.rx_enabled_channels = [0]  # Use only one RX channel

# --- Transmit the Query Command ---

# Start TX in full-duplex mode: TX channel 0 transmits the query waveform and channel 1 is silent.
sdr.tx_cyclic_buffer = False
sdr.tx([waveform, np.zeros_like(waveform)])

# Calculate the duration of the query command.
query_duration = len(waveform) / SAMPLE_RATE

# Wait for the entire query command to be transmitted.
#time.sleep(query_duration)

# Wait an additional 500 microseconds for the tag to begin responding.
time.sleep(0.0005)

# Now capture the maximum allowed window (32768 samples ≈ 3.3 ms at 10 MSPS) to ensure the backscatter is included.
max_samples = 32768
raw_response = sdr.rx()[:max_samples]

# Stop TX transmission.
sdr.tx_destroy_buffer()


# --- Plot the Captured Tag Response ---
plot_signal("Tag Response Signal (Time Domain)", raw_response, SAMPLE_RATE)
plot_frequency_domain(raw_response, SAMPLE_RATE)
print(num_samples)
