import numpy as np
import adi
import time
from scipy.signal import find_peaks

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

def transmit_query_pluto(query_bits, center_freq=915e6, sample_rate=1e6, gain=-10):
    """
    Transmits an EPC Gen2 Query command using PlutoSDR.

    Args:
        query_bits (list): The binary Query command.
        center_freq (float): Transmission frequency in Hz (default 915 MHz).
        sample_rate (float): SDR sample rate in Hz (default 1 MHz).
        gain (int): TX gain in dB (default -10 dB).
    """
    # Initialize PlutoSDR
    sdr = adi.ad9361(uri='ip:192.168.2.1')

    # Configure PlutoSDR
    sdr.tx_lo = int(center_freq)  # Set transmission frequency (915 MHz for UHF RFID)
    sdr.tx_hardwaregain_chan0 = gain  # Adjust TX gain
    sdr.sample_rate = int(sample_rate)  # Set sample rate

    # Generate the PIE waveform for the Query command
    waveform = encode_pie(query_bits, sample_rate)

    # Transmit the waveform
    sdr.tx_cyclic_buffer = True  # Enable cyclic transmission
    sdr.tx([waveform, waveform])  # Send waveform (both channels enabled)

    # Allow transmission for a brief period
    time.sleep(0.05)  # Transmit for 50ms
    sdr.tx_destroy_buffer()  # Stop transmission

    print("Query command transmitted.")


def receive_rn16_pluto(center_freq=915e6, sample_rate=1e6, gain=40, capture_time=0.02):
    """
    Receives the RN16 response from an RFID tag using PlutoSDR.

    Args:
        center_freq (float): Frequency in Hz (default 915 MHz).
        sample_rate (float): SDR sample rate (default 1 MHz).
        gain (int): RX gain in dB (default 40).
        capture_time (float): Duration to capture signal in seconds (default 20 ms).

    Returns:
        np.array: Captured raw signal samples.
    """
    # Initialize PlutoSDR for receiving
    sdr = adi.ad9361(uri='ip:192.168.2.1')

    # Configure SDR settings
    sdr.rx_lo = int(center_freq)  # Set receive frequency (915 MHz for UHF RFID)
    sdr.sample_rate = int(sample_rate)  # Set sample rate
    sdr.rx_hardwaregain_chan0 = gain  # Adjust receive gain
    sdr.rx_enabled_channels = [0]  # TODO only using one channel for now

    # Capture raw signal
    num_samples = int(sample_rate * capture_time)  # Total samples to capture
    raw_signal = sdr.rx()[:num_samples]  # Receive data

    print("RN16 Response Captured!")
    return raw_signal

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

def fm0_decode(signal, sample_rate, bit_rate=64e3, threshold=None):
    """
    FM0 decodes the received RFID RN16 or EPC response.

    Args:
        signal (np.array): Received signal samples.
        sample_rate (float): Sample rate in Hz.
        bit_rate (float): Expected RFID bit rate in Hz (default 64 kbps).
        threshold (float, optional): Threshold for signal binarization.

    Returns:
        list: Decoded binary sequence.
    """
    # Detect preamble to align decoding
    preamble_start = detect_preamble(signal, sample_rate, bit_rate, threshold)
    if preamble_start == -1:
        print("Preamble not found. Decoding may be inaccurate.")
        preamble_start = 0  # Default to start of signal if preamble is missing

    # Auto-thresholding if not provided
    if threshold is None:
        threshold = (np.max(signal) + np.min(signal)) / 2

    # Convert signal to binary
    binary_signal = (signal > threshold).astype(int)

    # Detect transitions using peak detection
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
            decoded_bits.append(previous_bit)  # Long duration means no mid-bit transition
        else:
            decoded_bits.append(1 - previous_bit)  # Short duration means mid-bit transition
        previous_bit = decoded_bits[-1]

    return decoded_bits


def extract_rn16(decoded_bits):
    """
    Extracts the RN16 and CRC from the FM0-decoded bits.

    Args:
        decoded_bits (list): Binary sequence from FM0 decoding.

    Returns:
        tuple: (RN16, CRC-16)
    """
    if len(decoded_bits) < 32:  # RN16 (16 bits) + CRC (16 bits)
        print("Error: Incomplete RN16 response!")
        return None, None

    rn16 = decoded_bits[:16]  # Extract RN16 (first 16 bits)
    crc = decoded_bits[16:32]  # Extract CRC-16 (next 16 bits)

    return rn16, crc

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

def validate_rn16(rn16, received_crc):
    """
    Validates RN16 by checking CRC-16.

    Args:
        rn16 (list): Extracted RN16 bits.
        received_crc (list): Received CRC bits.

    Returns:
        bool: True if CRC matches, False otherwise.
    """
    computed_crc = compute_crc16(rn16)
    return computed_crc == received_crc


def generate_ack_command(rn16):
    """
    Generates the 18-bit ACK command for RFID communication.

    Args:
        rn16 (list): The validated 16-bit RN16 binary list.

    Returns:
        list: The 18-bit ACK command binary list.
    """
    ack_prefix = [0, 1]  # 2-bit ACK command header
    return ack_prefix + rn16  # Concatenate header with RN16

def encode_pie(command_bits, sample_rate=1e6, tari=12.5e-6):
    """
    Encodes an EPC Gen2 command using Pulse Interval Encoding (PIE).

    Args:
        command_bits (list): The binary command to encode.
        sample_rate (float): SDR sample rate in Hz (default 1 MHz).
        tari (float): Reference time interval (default 12.5 µs).

    Returns:
        np.array: Encoded PIE waveform (ASK-modulated).
    """
    short_pulse = int(sample_rate * tari)  # Short pulse duration
    long_pulse = 2 * short_pulse  # Long pulse is twice short pulse

    waveform = []
    for bit in command_bits:
        if bit == 0:
            waveform.extend([1.0] * short_pulse)  # Short high
            waveform.extend([0.0] * long_pulse)  # Long low
        else:
            waveform.extend([1.0] * long_pulse)  # Long high
            waveform.extend([0.0] * short_pulse)  # Short low

    return np.array(waveform, dtype=np.complex64)

def transmit_ack_pluto(rn16, center_freq=915e6, sample_rate=1e6, gain=-10):
    """
    Transmits an EPC Gen2 ACK command using PlutoSDR.

    Args:
        rn16 (list): The validated 16-bit RN16.
        center_freq (float): Transmission frequency in Hz (default 915 MHz).
        sample_rate (float): SDR sample rate in Hz (default 1 MHz).
        gain (int): TX gain in dB (default -10 dB).
    """
    # Initialize PlutoSDR
    sdr = adi.ad9361(uri='ip:192.168.2.1')

    # Configure PlutoSDR
    sdr.tx_lo = int(center_freq)  # Set center frequency (915 MHz)
    sdr.tx_hardwaregain_chan0 = gain  # Adjust TX gain
    sdr.sample_rate = int(sample_rate)  # Set sample rate

    # Generate ACK command
    ack_command = generate_ack_command(rn16)

    # Encode ACK command using PIE
    waveform = encode_pie(ack_command, sample_rate)

    # Transmit the waveform
    sdr.tx_cyclic_buffer = True  # Enable cyclic transmission
    sdr.tx([waveform, waveform])  # Send waveform

    # Allow transmission for a brief period
    time.sleep(0.05)  # Transmit for 50ms
    sdr.tx_destroy_buffer()  # Stop transmission

    print("ACK command transmitted.")

def receive_epc_pluto(center_freq=915e6, sample_rate=1e6, gain=40, capture_time=0.02):
    """
    Receives the EPC response from an RFID tag using PlutoSDR.

    Args:
        center_freq (float): Frequency in Hz (default 915 MHz).
        sample_rate (float): SDR sample rate (default 1 MHz).
        gain (int): RX gain in dB (default 40).
        capture_time (float): Duration to capture signal in seconds (default 20 ms).

    Returns:
        np.array: Captured raw signal samples.
    """
    # Initialize PlutoSDR
    sdr = adi.ad9361(uri='ip:192.168.2.1')

    # Configure SDR settings
    sdr.rx_lo = int(center_freq)  # Set receive frequency (915 MHz)
    sdr.sample_rate = int(sample_rate)  # Set sample rate
    sdr.rx_hardwaregain_chan0 = gain  # Adjust receive gain
    sdr.rx_enabled_channels = [0]  # TODO only using one channel for now

    # Capture raw signal
    num_samples = int(sample_rate * capture_time)  # Total samples to capture
    raw_signal = sdr.rx()[:num_samples]  # Receive data

    print("EPC Response Captured!")
    return raw_signal

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

# **Main Execution:**

# Step 1: Transmit query command
query_command = [1, 0, 1, 1, 0, 0, 1, 0]  # Example Query command
transmit_query_pluto(query_command)

# Step 2: Capture RN16 response
raw_signal = receive_rn16_pluto()

# Step 3: FM0 Decode the received signal
decoded_bits = fm0_decode(raw_signal, sample_rate=1e6)

# Step 4: Extract RN16 and CRC
rn16, crc = extract_rn16(decoded_bits)

if rn16:
    print("Extracted RN16:", ''.join(map(str, rn16)))

    # Step 5: Validate RN16 using CRC
    if validate_rn16(rn16, crc):
        print("✅ RN16 is valid!")
    else:
        print("❌ CRC check failed. Data may be corrupted.")

# Step 6: Transmit ACK command
transmit_ack_pluto(rn16)

# Step 7: Capture EPC response
raw_signal = receive_epc_pluto()

# Step 8: FM0 Decode the received signal
decoded_bits = fm0_decode(raw_signal, sample_rate=1e6)

# Step 9: Extract EPC components
pc, epc, crc = extract_epc(decoded_bits)

if epc:
    print("Extracted EPC:", ''.join(map(str, epc)))

    # Step 10: Validate EPC using CRC
    if validate_epc(pc + epc, crc):
        print("✅ EPC is valid!")
    else:
        print("❌ CRC check failed. Data may be corrupted.")