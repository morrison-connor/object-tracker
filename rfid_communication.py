import adi
import numpy as np

# Create radio
sdr = adi.ad9361(uri='ip:192.168.2.1')

# Properties (taken directly from example)
sdr.rx_rf_bandwidth = int(30e6)
sdr.sample_rate = 6e6
sdr.rx_lo = int(900e6) # carrier frequency of the RX path
sdr.tx_lo = int(900e6)
sdr.tx_cyclic_buffer = False
sdr.tx_hardwaregain_chan0 = -3
sdr.gain_control_mode_chan0 = "manual"

# Configure data channels
sdr.rx_enabled_channels = [0, 1]
#sdr.tx_enabled_channels = [0]

# Create output waveform
fs = int(sdr.sample_rate)
N = 2**10
#fc = int(1e6 / (fs / N) * (fs / N))
fc0 = int(200e3)
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr.tx([iq0,iq0])  # Send Tx data.

# Read data
data = sdr.rx()
rx_0 = data[0]
rx_1 = data[1]