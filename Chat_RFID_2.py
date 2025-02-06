import numpy as np
import adi
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

global debug
debug = False


def crc5(bits):
    """
    Computes CRC-5 checksum for the EPC Gen2 Query command.
    
    Args:
        bits (list): The Query command bits before CRC is appended.
    
    Returns:
        list: The 5-bit CRC as a list of integers.
    """
    poly = 0b101001  # EPC Gen2 CRC-5 Polynomial (x^5 + x^3 + 1)
    crc = 0b11111  # Initial CRC value per EPC Gen2 spec
    
    for bit in bits:
        crc_in = (crc >> 4) & 1  # Extract leftmost bit
        crc = ((crc << 1) & 0b11111) | bit  # Shift and add new bit
        if crc_in:  
            crc ^= poly  # XOR with polynomial if MSB was 1

    return [(crc >> i) & 1 for i in reversed(range(5))]  # Return as 5-bit list

def build_query_command(dr=1, M=2, TRext=1, Sel=0, Session=0, Target=0, Q=4):
    """
    Constructs a complete EPC Gen2 Query command with CRC-5, allowing Miller encoding.

    Args:
        dr (int): Divide Ratio (1 bit, 0 = DR=64/3, 1 = DR=8)
        M (int): **Miller encoding cycles per symbol** (2 bits: 00=1, 01=2, 10=4, 11=8)
        TRext (int): TRext flag (1 bit, 0 = standard preamble, 1 = extended preamble)
        Sel (int): Tag selection flag (2 bits)
        Session (int): Inventory session (2 bits)
        Target (int): Target flag (1 bit, 0=A, 1=B) - tags will only respond if their A/B state matches the query A/B state
        Q (int): Slot-count parameter (4 bits, 0-15) - variable to control number of possible tag responses

    Returns:
        list: EPC Gen2 Query command bit sequence including preamble.
    """
    if M not in [1, 2, 4, 8]:
        raise ValueError("M must be 1 (FM0), 2, 4, or 8 (Miller encoding).")

    # Convert Miller value to correct bit encoding (per EPC Gen2 spec)
    miller_map = {1: [0, 0], 2: [0, 1], 4: [1, 0], 8: [1, 1]}
    M_bits = miller_map[M]

    # Step 1: Construct Query command fields (excluding CRC)
    query_bits = [1, 0, 0, 0]  # Command Code: 4 bits (Fixed as '1000')
    query_bits.append(dr)  # Divide Ratio (1 bit)
    query_bits.extend(M_bits)  # **Miller Encoding Field (2 bits)**
    query_bits.append(TRext)  # TRext (1 bit)
    query_bits.extend([(Sel >> 1) & 1, Sel & 1])  # Sel (2 bits)
    query_bits.extend([(Session >> 1) & 1, Session & 1])  # Session (2 bits)
    query_bits.append(Target)  # Target (1 bit)
    query_bits.extend([(Q >> 3) & 1, (Q >> 2) & 1, (Q >> 1) & 1, Q & 1])  # Q (4 bits)

    # Step 2: Compute CRC-5 and append
    crc_bits = crc5(query_bits)
    query_bits.extend(crc_bits)

    # Step 3: Prepend preamble (depends on TRext)
    if TRext == 0:
        preamble = [0] * 12 + [1, 0, 1, 0]  # Standard preamble (12 zeros + violation bit)
    else:
        preamble = [1] * 12 + [1, 0, 1, 0]  # Extended preamble with pilot tone

    full_bits = preamble + query_bits  # Full Query command

    return full_bits  # Ready for PIE encoding

def transmit_query_pluto(query_bits, center_freq=915e6, sample_rate=10e6, gain=-10):
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
    sdr.sample_rate = int(sample_rate)  # Set sample rate
    sdr.tx_lo = int(center_freq)  # Set transmission frequency (915 MHz for UHF RFID)
    sdr.tx_hardwaregain_chan0 = gain  # Adjust TX gain
    sdr.tx_cyclic_buffer = True  # Enable cyclic transmission
    # TODO potentially remove cyclic buffer - Chat says this may stretch signal

    # Generate the PIE waveform for the Query command
    waveform = encode_pie(query_bits, sample_rate)
    t = np.arange(len(waveform)) / sample_rate
    carrier_frequency = 100e3  # Subcarrier freq, PlutoSDR will upconvert to tx_lo value
    modulated_waveform = waveform * np.cos(2 * np.pi * carrier_frequency * t)

    if debug:
        plot_generic_signal("PIE Encoded Query TX", modulated_waveform)

    # TODO test if we need to do modulation?

    # Transmit the waveform
    sdr.tx([modulated_waveform, modulated_waveform])  # Send waveform (both channels enabled)

    # Allow transmission for a brief period
    
    time.sleep(0.1)  # Transmit for 100ms    
    sdr.tx_destroy_buffer()  # Stop transmission

    print("Query command transmitted.")


def receive_rn16_pluto(center_freq=915e6, sample_rate=10e6, gain=20, capture_time=0.02):
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

def detect_preamble(decoded_bits):
    """
    Detects the FM0 preamble in the decoded binary sequence.

    Args:
        decoded_bits (list): The binary sequence after thresholding.

    Returns:
        int: Index of the preamble start, or -1 if not found.
    """

    # TODO is this the correct preamble pattern?
    preamble = [1, 0, 1, 0]  # FM0 preamble pattern
    for i in range(len(decoded_bits) - len(preamble) + 1):
        if decoded_bits[i:i+4] == preamble:
            return i + 4  # Return start of RN16 (after preamble)
    return -1  # Preamble not found

def fm0_decode(signal, sample_rate, bit_rate=40e3, threshold=None):
    """
    FM0 decodes an RFID RN16 response after preamble detection.

    Args:
        signal (np.array): Received signal samples.
        sample_rate (float): Sample rate in Hz.
        bit_rate (float): Expected RFID bit rate in Hz (default 64 kbps).
        threshold (float, optional): Threshold for signal binarization.

    Returns:
        list: Decoded binary sequence.
    """
    demod_signal = np.abs(signal)
    if threshold is None:
        threshold = (np.max(demod_signal) + np.min(demod_signal)) / 2

    binary_signal = (demod_signal > threshold).astype(int)

    bit_period = int(sample_rate / bit_rate)
    peaks, _ = find_peaks(binary_signal, distance=bit_period//2)
    valleys, _ = find_peaks(-binary_signal, distance=bit_period//2)
    transitions = np.sort(np.concatenate((peaks, valleys)))

    # Detect all transitions
    transitions = np.where(np.diff(binary_signal) != 0)[0]

    decoded_bits = []
    previous_bit = 1  # Start assumption

    for i in range(len(transitions) - 1):
        duration = transitions[i + 1] - transitions[i]
        if duration >= bit_period:
            decoded_bits.append(previous_bit)  # Long period = same bit
        else:
            decoded_bits.append(1 - previous_bit)  # Short period = bit flip
        previous_bit = decoded_bits[-1]

    return decoded_bits

def extract_rn16(decoded_bits):
    """
    Extracts RN16 and CRC from the decoded bits after preamble alignment.

    Args:
        decoded_bits (list): FM0-decoded binary sequence.

    Returns:
        tuple: (RN16, CRC-16)
    """
    preamble_start = detect_preamble(decoded_bits)
    if preamble_start == -1:
        print("Error: Preamble not found!")
        return None, None

    rn16 = decoded_bits[preamble_start:preamble_start + 16]
    crc16 = decoded_bits[preamble_start + 16:preamble_start + 32]

    return rn16, crc16

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
        # Shift the CRC left by 1 and bring in the new bit.
        msb = (crc >> 15) & 1
        crc = ((crc << 1) & 0xFFFF) | bit
        # If the shifted-out bit (msb) was different from the incoming bit,
        # perform the XOR with the polynomial.
        if msb:
            crc ^= poly
    # Return the CRC as a list of 16 bits.
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

import numpy as np

def encode_pie(command_bits, sample_rate=10e6, bitrate=40e3, high=2**14, low=0):
    """
    Encodes an EPC Gen2 command using Pulse Interval Encoding (PIE).

    Args:
        command_bits (list): The binary command to encode.
        sample_rate (float): SDR sample rate in Hz (default 1 MHz).
        bitrate (float): Bitrate of the tag (default 40 kbps).
        high (int): High signal level for ASK modulation.
        low (int): Low signal level for ASK modulation.

    Returns:
        np.array: Encoded PIE waveform (ASK-modulated).
    """
    tari = 1 / bitrate  # Base unit time (Tari) based on tag bitrate
    short_pulse = int(sample_rate * tari)  # Short high time
    long_pulse = int(1.5 * sample_rate * tari)  # Long low time

    waveform = []
    for bit in command_bits:
        if bit == 0:
            waveform.extend([high] * short_pulse)  # Short high (Tari)
            waveform.extend([low] * long_pulse)    # Long low (1.5 * Tari)
        else:
            waveform.extend([high] * (2 * short_pulse))  # Long high (2 * Tari)
            waveform.extend([low] * short_pulse)         # Short low (Tari)

    return np.array(waveform, dtype=np.float32)  # Ensure real output for ASK


def transmit_ack_pluto(rn16, center_freq=915e6, sample_rate=10e6, gain=-10):
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
    time.sleep(0.1)  # Transmit for 100ms
    sdr.tx_destroy_buffer()  # Stop transmission

    print("ACK command transmitted.")

def receive_epc_pluto(center_freq=915e6, sample_rate=10e6, gain=20, capture_time=0.02):
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

def plot_generic_signal(title, signal, sample_rate=10e6):
    t = np.linspace(0, (len(signal) - 1) / sample_rate, len(signal))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, color="purple")
    plt.title(title + " - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    freq = np.fft.fftfreq(len(signal), d=1/sample_rate)
    fft_signal = np.fft.fft(signal)
    plt.subplot(2, 1, 2)
    plt.plot(freq[:int(sample_rate)//2], np.abs(fft_signal[:int(sample_rate)//2]))
    plt.title(title + " - Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_received_signal(signal, sample_rate, bit_rate=40e3):
    """
    Plots the raw RFID signal, thresholded binary signal, and detected transitions.

    Args:
        signal (np.array): The raw received signal.
        sample_rate (float): The SDR sample rate.
        bit_rate (float): Expected RFID bit rate (default 64 kbps).
    """
    # Convert complex signal to amplitude
    signal_amplitude = np.abs(signal)

    # Auto-thresholding
    threshold = (np.max(signal_amplitude) + np.min(signal_amplitude)) / 2
    binary_signal = (signal_amplitude > threshold).astype(int)

    # Detect transitions
    bit_period = int(sample_rate / bit_rate)
    peaks, _ = find_peaks(binary_signal, distance=bit_period//2)
    valleys, _ = find_peaks(-binary_signal, distance=bit_period//2)
    transitions = np.sort(np.concatenate((peaks, valleys)))

    # Plot raw signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal_amplitude, label="Raw Signal", color="blue", alpha=0.7)
    plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
    plt.scatter(transitions, signal_amplitude[transitions], color='green', marker='o', label="Transitions")
    plt.title("Raw RFID Signal (Amplitude)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Plot thresholded binary signal
    plt.subplot(2, 1, 2)
    plt.plot(binary_signal, label="Binary Signal", drawstyle="steps-pre", color="black")
    plt.scatter(transitions, binary_signal[transitions], color='red', marker='o', label="Transitions")
    plt.title("Thresholded RFID Signal (Binary)")
    plt.xlabel("Sample Index")
    plt.ylabel("Binary Value")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# **Main Execution:**

# Step 1: Transmit query command
query_command = build_query_command(dr=1, M=4, TRext=0, Sel=0, Session=1, Target=0, Q=4)
print("Query Command with Miller M=4:", query_command)
transmit_query_pluto(query_command)

# Step 2: Capture RN16 response
raw_signal = receive_rn16_pluto()
if debug:
    plot_received_signal(raw_signal, sample_rate=10e6)

# Step 3: FM0 Decode the received signal
decoded_bits = fm0_decode(raw_signal, sample_rate=10e6)

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
decoded_bits = fm0_decode(raw_signal, sample_rate=10e6)

# Step 9: Extract EPC components
pc, epc, crc = extract_epc(decoded_bits)

if epc:
    print("Extracted EPC:", ''.join(map(str, epc)))

    # Step 10: Validate EPC using CRC
    if validate_epc(pc + epc, crc):
        print("✅ EPC is valid!")
    else:
        print("❌ CRC check failed. Data may be corrupted.")