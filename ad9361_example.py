# Copyright (C) 2022 Analog Devices, Inc.
#
# SPDX short identifier: ADIBSD

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import adi


def plot_generic_signal(title, signal, sample_rate=30e6):
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

# Create radio
sdr = adi.ad9361(uri='ip:192.168.2.1')

# Configure properties
sdr.rx_rf_bandwidth = 4000000
sdr.sample_rate = 6000000
sdr.rx_lo = 2000000000
sdr.tx_lo = 2000000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -30
sdr.gain_control_mode_chan0 = "slow_attack"

# Configuration data channels
sdr.rx_enabled_channels = [0]
sdr.tx_enabled_channels = [0]

# Read properties
print("RX LO %s" % (sdr.rx_lo))

# Create a sinewave waveform
fs = int(sdr.sample_rate)
N = 1024
fc = int(1000000 / (fs / N)) * (fs / N)
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = i + 1j * q

plot_generic_signal("Sin wave", iq)

time.sleep(2)

# Send data
sdr.tx(iq)

time.sleep(2)

# Collect data
for r in range(20):
    x = sdr.rx()
    f, Pxx_den = signal.periodogram(x, fs)
    plt.clf()
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.draw()
    plt.pause(0.05)
    time.sleep(0.1)

#plt.show()