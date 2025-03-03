import numpy as np
import matplotlib.pyplot as plt
import adi
import time
from scipy.signal import butter, filtfilt, find_peaks


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


def encode_pie(query_bits, sample_rate=1e6, tari=12.5e-6):
    """
    Encodes an EPC Gen2 Query command using Pulse Interval Encoding (PIE).

    Args:
        query_bits (list): The binary Query command to encode.
        sample_rate (float): The SDR sample rate in Hz (default 1 MHz).
        tari (float): Reference time interval (default 12.5 µs).

    Returns:
        np.array: Encoded waveform (ASK-modulated).
    """
    short_pulse = int(sample_rate * tari)  # Short duration for PIE
    long_pulse = 2 * short_pulse  # Long duration is twice short duration

    waveform = []

    for bit in query_bits:
        if bit == 0:
            # Data-0: Short high, Long low
            waveform.extend([1.0] * short_pulse)  # High pulse
            waveform.extend([0.0] * long_pulse)  # Low pulse
        else:
            # Data-1: Long high, Short low
            waveform.extend([1.0] * long_pulse)  # High pulse
            waveform.extend([0.0] * short_pulse)  # Low pulse

    # Convert to complex format for PlutoSDR transmission
    return np.array(waveform, dtype=np.complex64)

def transmit_query_pluto(sdr, query_bits, center_freq=915e6, sample_rate=1e6, gain=-10):
    """
    Transmits an EPC Gen2 Query command using PlutoSDR.

    Args:
        query_bits (list): The binary Query command.
        center_freq (float): Transmission frequency in Hz (default 915 MHz).
        sample_rate (float): SDR sample rate in Hz (default 1 MHz).
        gain (int): TX gain in dB (default -10 dB).
    """
    # Initialize PlutoSDR
    #sdr = adi.Pluto("ip:192.168.2.1")  # Update with actual Pluto IP if needed

    # Configure PlutoSDR
    sdr.tx_lo = int(center_freq)  # Set transmission frequency (915 MHz for UHF RFID)
    sdr.tx_hardwaregain_chan0 = gain  # Adjust TX gain
    sdr.sample_rate = int(sample_rate)  # Set sample rate

    # Generate the PIE waveform for the Query command
    waveform = encode_pie(query_bits, sample_rate)

    # Transmit the waveform
    sdr.tx_cyclic_buffer = True  # Enable cyclic transmission
    sdr.tx([waveform, waveform])  # Send waveform

    # Allow transmission for a brief period
    time.sleep(0.05)  # Transmit for 50ms
    sdr.tx_destroy_buffer()  # Stop transmission

    print("Query command transmitted.")


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
    sdr.sample_rate = int(fs)  # Set sample rate
    sdr.tx_lo = int(carrier_freq)
    sdr.tx_hardwaregain_chan0 = -10
    sdr.tx([signal, signal])

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Creates a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """Applies a bandpass filter to remove noise and unwanted frequencies."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)

def envelope_detection(signal):
    """Extracts the envelope of the received signal using Hilbert transform."""
    #generic_signal_plot("pre envelope", signal)
    analytic_signal = np.abs(np.fft.ifft(np.fft.fft(signal)))
    #generic_signal_plot("post envelope", signal)
    return analytic_signal

def old_fm0_decode(received_signal, fs, bit_rate):
    """Decodes an FM0 encoded RFID response to recover binary data."""

    # TODO entire function probably wrong

    symbol_length = int(fs / bit_rate)
    threshold = np.mean(received_signal)
    bits = (received_signal > threshold).astype(int)
    decoded_bits = []
    for i in range(0, len(bits), symbol_length):
        decoded_bits.append(bits[i])
    return decoded_bits


def detect_preamble(signal, sampling_rate, bit_rate, threshold=None):
    """
    Detects the FM0 preamble in an RFID response.

    Args:
        signal (np.array): The received signal (amplitude values).
        sampling_rate (float): The sample rate of the SDR in Hz.
        bit_rate (float): The expected RFID bit rate in Hz.
        threshold (float, optional): Threshold for distinguishing between high and low states. Auto-calculated if None.

    Returns:
        int: Index of the preamble start, or -1 if not found.
    """
    # Normalize and threshold signal
    signal = np.array(signal)
    if threshold is None:
        threshold = (np.max(signal) + np.min(signal)) / 2
    binary_signal = (signal > threshold).astype(int)

    # Expected FM0 preamble pattern for EPC Gen2 (example: 1010 or 1100)
    expected_preamble = [1, 0, 1, 0]  # Adjust if needed

    # Detect transitions using peak detection
    bit_period = int(sampling_rate / bit_rate)
    peaks, _ = find_peaks(binary_signal, distance=bit_period//2)
    valleys, _ = find_peaks(-binary_signal, distance=bit_period//2)
    transitions = np.sort(np.concatenate((peaks, valleys)))

    # Try to match the expected preamble pattern
    for i in range(len(transitions) - len(expected_preamble) + 1):
        segment = binary_signal[transitions[i]:transitions[i + len(expected_preamble)]]
        if list(segment[:len(expected_preamble)]) == expected_preamble:
            return transitions[i]  # Return start index of preamble

    return -1  # Preamble not found


def decode_fm0(signal, sampling_rate, bit_rate, threshold=None):
    """
    Decodes an FM0-encoded RFID response.

    Args:
        signal (np.array): The received signal (amplitude values).
        sampling_rate (float): The sample rate of the SDR in Hz.
        bit_rate (float): The expected RFID bit rate in Hz.
        threshold (float, optional): Threshold for distinguishing between high and low states. If None, auto-calculated.

    Returns:
        list: Decoded binary sequence.
    """

    # Normalize the signal
    signal = np.array(signal)
    if threshold is None:
        threshold = (np.max(signal) + np.min(signal)) / 2  # Adaptive thresholding

    binary_signal = (signal > threshold).astype(int)

    # Detect bit transitions (peaks and valleys)
    bit_period = int(sampling_rate / bit_rate)
    peaks, _ = find_peaks(binary_signal, distance=bit_period//2)  # Rising edges
    valleys, _ = find_peaks(-binary_signal, distance=bit_period//2)  # Falling edges

    # Merge peaks and valleys to get all transitions
    transitions = np.sort(np.concatenate((peaks, valleys)))

    decoded_bits = []
    previous_bit = 1  # FM0 starts with an assumed "1"

    for i in range(len(transitions) - 1):
        duration = transitions[i + 1] - transitions[i]
        if duration >= bit_period:  # A long duration means no mid-bit transition → "1"
            decoded_bits.append(previous_bit)
        else:  # A mid-bit transition → "0"
            decoded_bits.append(1 - previous_bit)
        previous_bit = decoded_bits[-1]

    return decoded_bits


def old_extract_rn16(decoded_bits):
    """Extracts the RN16 data from the decoded tag response."""
    preamble = [1, 0, 1, 0]
    index = -1
    for i in range(len(decoded_bits) - len(preamble)):
        if decoded_bits[i:i+len(preamble)] == preamble:
            index = i + len(preamble)
            break
    if index == -1:
        print("Preamble not found")
        return None
    return decoded_bits[index:index+16]  # RN16 (first response from the tag)


def extract_rn16(decoded_bits):
    """
    Extracts the RN16 from the decoded FM0 binary sequence.

    Args:
        decoded_bits (list): The binary sequence after FM0 decoding.

    Returns:
        tuple: (RN16 binary list, CRC-16 binary list)
    """
    # EPC Gen2 FM0 preamble is typically 4 bits: "1010"
    preamble = [1, 0, 1, 0]

    # Locate the preamble in the decoded bits
    for i in range(len(decoded_bits) - 20):  # 4 (preamble) + 16 (RN16)
        if decoded_bits[i:i+4] == preamble:
            rn16_start = i + 4  # Start after preamble
            rn16 = decoded_bits[rn16_start:rn16_start+16]
            crc16 = decoded_bits[rn16_start+16:rn16_start+32]  # Next 16 bits

            if validate_rn16(rn16, crc16):
                return rn16, crc16

    return None, None  # If no RN16 is found

def compute_crc16(data_bits):
    """
    Computes the CRC-16 (EPC Gen2) checksum for a given binary sequence.
    
    Args:
        data_bits (list): The binary sequence to compute CRC on (should be 16-bit RN16).

    Returns:
        list: The 16-bit CRC result as a list of binary values.
    """
    crc = 0xFFFF  # EPC Gen2 CRC-16 starts with 0xFFFF
    poly = 0x1021  # CRC polynomial x^16 + x^12 + x^5 + 1

    for bit in data_bits:
        crc ^= (bit << 15)  # XOR the input bit with the highest bit of CRC
        for _ in range(8):  # Process 8 cycles per bit
            if crc & 0x8000:  # If MSB is 1, shift left and XOR with polynomial
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF  # Keep CRC within 16-bit range

    # Convert CRC result to a binary list
    return [int(b) for b in f"{crc:016b}"]  # Convert to 16-bit binary array

def validate_rn16(rn16, received_crc):
    """
    Validates the RN16 response by comparing the computed CRC with the received CRC.

    Args:
        rn16 (list): The 16-bit RN16 binary list.
        received_crc (list): The received 16-bit CRC binary list.

    Returns:
        bool: True if CRC matches, False if the data is corrupted.
    """
    computed_crc = compute_crc16(rn16)
    return computed_crc == received_crc  # Compare computed and received CRC


def receive_and_process_signal(sdr, fs, bit_rate):
    """Receives and processes RFID signals to identify EPC Gen2 tags."""
    raw_signal = sdr.rx()
    filtered_signal = apply_bandpass_filter(raw_signal, 100e3, 400e3, fs)
    envelope = envelope_detection(filtered_signal)
    decoded_bits = decode_fm0(envelope, fs, bit_rate)
    rn_16 = extract_rn16(decoded_bits)
    return rn_16


def generate_ack_command(rn16):
    """
    Generates the ACK command for RFID communication.

    Args:
        rn16 (list): The validated 16-bit RN16 binary list.

    Returns:
        list: The 18-bit ACK command binary list.
    """
    ack_prefix = [0, 1]  # 2-bit ACK command header
    return ack_prefix + rn16  # Concatenate header with RN16


def generate_ack_waveform(ack_bits, sample_rate=1e6, tari=12.5e-6):
    """
    Generates an ASK-modulated waveform for an RFID ACK command.

    Args:
        ack_bits (list): The 18-bit ACK command in binary.
        sample_rate (float): SDR sample rate in Hz (default 1 MHz).
        tari (float): Tari (reference time interval) in seconds (default 12.5 µs).

    Returns:
        np.array: ASK-modulated waveform (complex values for PlutoSDR).
    """
    bit_duration = tari  # Duration of each bit in seconds
    samples_per_bit = int(sample_rate * bit_duration)  # Samples per bit

    waveform = []
    for bit in ack_bits:
        if bit == 1:
            symbol = np.ones(samples_per_bit)  # High (carrier on)
        else:
            symbol = np.zeros(samples_per_bit)  # Low (carrier off)
        waveform.extend(symbol)

    # Convert to complex format for PlutoSDR transmission
    return np.array(waveform, dtype=np.complex64)


def receive_epc_pluto(center_freq=915e6, sample_rate=1e6, gain=40, capture_time=0.02):
    """
    Receives an RFID EPC response from the tag using PlutoSDR.

    Args:
        center_freq (float): Frequency in Hz (default 915 MHz).
        sample_rate (float): SDR sample rate (default 1 MHz).
        gain (int): RX gain in dB (default 40).
        capture_time (float): Duration to capture signal in seconds (default 20 ms).

    Returns:
        np.array: Captured raw signal samples.
    """
    # Initialize PlutoSDR
    sdr = adi.Pluto("ip:192.168.2.1")  # Update with actual Pluto IP if needed

    # Configure PlutoSDR for reception
    sdr.rx_lo = int(center_freq)  # Set center frequency
    sdr.sample_rate = int(sample_rate)  # Sample rate
    sdr.rx_hardwaregain_chan0 = gain  # Adjust RX gain

    # Capture raw signal
    num_samples = int(sample_rate * capture_time)
    raw_signal = sdr.rx()[:num_samples]  # Receive raw samples

    print("EPC Response Captured!")
    return raw_signal

def fm0_decode(signal, sample_rate, bit_rate=64000, threshold=None):
    """
    FM0 decodes the received RFID response.

    Args:
        signal (np.array): Received signal samples.
        sample_rate (float): Sample rate in Hz.
        bit_rate (float): Expected RFID bit rate in Hz (default 64 kbps).
        threshold (float, optional): Threshold for signal binarization.

    Returns:
        list: Decoded binary sequence.
    """
    # Auto-thresholding if not provided
    if threshold is None:
        threshold = (np.max(signal) + np.min(signal)) / 2

    # Convert to binary
    binary_signal = (signal > threshold).astype(int)

    # Detect bit transitions using peak detection
    bit_period = int(sample_rate / bit_rate)
    peaks, _ = find_peaks(binary_signal, distance=bit_period//2)
    valleys, _ = find_peaks(-binary_signal, distance=bit_period//2)
    transitions = np.sort(np.concatenate((peaks, valleys)))

    # Decode FM0
    decoded_bits = []
    previous_bit = 1  # FM0 always starts with a known state

    for i in range(len(transitions) - 1):
        duration = transitions[i + 1] - transitions[i]
        if duration >= bit_period:
            decoded_bits.append(previous_bit)
        else:
            decoded_bits.append(1 - previous_bit)
        previous_bit = decoded_bits[-1]

    return decoded_bits

def extract_epc(decoded_bits):
    """
    Extracts the EPC from the FM0-decoded bits.

    Args:
        decoded_bits (list): Binary sequence from FM0 decoding.

    Returns:
        tuple: (PC, EPC, CRC)
    """
    if len(decoded_bits) < 128:  # Ensure we have enough bits
        print("Error: Incomplete EPC response!")
        return None, None, None

    # EPC Gen2 Format: PC (16 bits) + EPC (96 bits) + CRC (16 bits)
    pc = decoded_bits[:16]  # Protocol Control (PC)
    epc = decoded_bits[16:112]  # EPC (typically 96 bits)
    crc = decoded_bits[112:128]  # CRC-16

    return pc, epc, crc

def compute_crc16(data_bits):
    """
    Computes CRC-16 for validation.

    Args:
        data_bits (list): Data bits to compute CRC on.

    Returns:
        list: Computed CRC-16.
    """
    crc = 0xFFFF  # EPC Gen2 CRC-16 starts at 0xFFFF
    poly = 0x1021  # x^16 + x^12 + x^5 + 1

    for bit in data_bits:
        crc ^= (bit << 15)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF  # Keep CRC in 16-bit range

    return [int(b) for b in f"{crc:016b}"]

def validate_epc(epc_bits, received_crc):
    """
    Validates EPC by checking CRC-16.

    Args:
        epc_bits (list): Extracted EPC bits.
        received_crc (list): Received CRC bits.

    Returns:
        bool: True if CRC matches, False otherwise.
    """
    computed_crc = compute_crc16(epc_bits)
    return computed_crc == received_crc


def generic_signal_plot(title, signal):
    t = np.linspace(0, (len(signal) - 1) / fs, len(signal))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, color="purple")
    plt.title(title + " - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    freq = np.fft.fftfreq(len(signal), d=1/fs)
    fft_signal = np.fft.fft(signal)
    plt.subplot(2, 1, 2)
    plt.plot(freq[:int(fs)//2], np.abs(fft_signal[:int(fs)//2]))
    plt.title(title + " - Frequency Domain")
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
#encoded_bits = miller_encode(query_cmd)
encoded_waveform = encode_pie(query_cmd)

# Display waveform (show only first 20 values for brevity)
print("Encoded PIE waveform:", encoded_waveform[:20])

# # Generate ASK waveform using encoded bits
# waveform = generate_waveform(encoded_bits, symbol_length)

# # Modulate waveform onto carrier
# # This step is handled by the AD9361 (assuming baseband mode) but is useful to visualize with plotting
# modulated_waveform, tx_t = modulate_waveform(waveform, carrier_freq, fs)

# # Convert to complex I/Q signal
# tx_signal = waveform + 1j * np.zeros_like(waveform)

# SDR setup
sdr = adi.ad9361(uri='ip:192.168.2.1')
#sdr.rx_buffer_size = int(symbol_length * len(encoded_bits) * 2)  # adjust dynamically based on cmd length

# Transmit signal
#transmit_signal(sdr, tx_signal, carrier_freq, fs)
#time.sleep(0.01)  # allow time for tag to respond

# Example usage:
query_command = [1, 0, 1, 1, 0, 0, 1, 0]  # Example Query command
transmit_query_pluto(sdr, query_command)


# Receive signals
sdr.rx_enabled_channels = [0]  # TODO only using one channel for now
sdr.rx_lo = int(carrier_freq)
sdr.rx_rf_bandwidth = int(fs)
rn16 = receive_and_process_signal(sdr, fs, bit_rate)
print("Detected RN16:", ''.join(map(str, rn16)))

ack_cmd = generate_ack_command(rn16)

sdr.tx_cyclic_buffer = True  # Enable continuous transmission

# Generate the ACK waveform
ack_waveform = generate_ack_waveform(ack_cmd, fs)

# transmit ACK waveform
transmit_signal(sdr, ack_waveform, carrier_freq, fs)

# Allow time for transmission before stopping
time.sleep(0.1)  # Transmit for 100ms
sdr.tx_destroy_buffer()  # Stop transmission

print("ACK command transmitted.")

# Main Execution:
raw_signal = receive_epc_pluto()  # Step 1: Capture EPC Response
decoded_bits = fm0_decode(raw_signal, sample_rate=1e6)  # Step 2: FM0 Decode
pc, epc, crc = extract_epc(decoded_bits)  # Step 3: Extract EPC Components

if epc:
    print("Extracted EPC:", ''.join(map(str, epc)))

    # Step 4: Validate EPC using CRC
    if validate_epc(pc + epc, crc):
        print("✅ EPC is valid!")
    else:
        print("❌ CRC check failed. Data may be corrupted.")



# def transmit_ack_pluto(ack_bits, center_freq, sample_rate, gain=-10):
#     """
#     Transmits an RFID ACK command using PlutoSDR.

#     Args:
#         ack_bits (list): The 18-bit ACK command in binary.
#         center_freq (float): Transmission frequency in Hz (default 915 MHz).
#         sample_rate (float): SDR sample rate in Hz (default 1 MHz).
#         gain (int): TX gain in dB (default -10 dB).

#     """
#     # Initialize PlutoSDR
#     sdr = adi.Pluto("ip:192.168.2.1")  # Update with actual IP if needed

#     # Configure PlutoSDR
#     sdr.tx_lo = int(center_freq)  # Set center frequency (e.g., 915 MHz for UHF RFID)
#     sdr.tx_hardwaregain_chan0 = gain  # Adjust transmission power
#     sdr.sample_rate = int(sample_rate)  # Set sample rate

#     # Generate the ASK waveform
#     waveform = generate_ack_waveform(ack_bits, sample_rate)

#     # Transmit the waveform
#     sdr.tx_cyclic_buffer = True  # Enable continuous transmission
#     sdr.tx(waveform)  # Send waveform

#     # Allow time for transmission before stopping
#     time.sleep(0.1)  # Transmit for 100ms
#     sdr.tx_destroy_buffer()  # Stop transmission

#     print("ACK command transmitted.")

