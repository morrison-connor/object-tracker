"""
Jon Kraft, Nov 5 2022
https://github.com/jonkraft/Pluto_Beamformer
video walkthrough of this at:  https://www.youtube.com/@jonkraft

"""
# Copyright (C) 2020 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Copyright (C) 2020 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import adi
import numpy as np
import pyqtgraph as pg   # pyqtgraph will plot MUCH faster than matplotlib (https://pyqtgraph.readthedocs.io/en/latest/getting_started/installation.html)
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import sys

DEBUG = False

'''User inputs'''
phase_cal = -172  # change this based on calibration to the phase shift value when AoA = 0
d_wavelength = 0.50  # distance between elements as a fraction of wavelength.  This is normally 0.5
phase_delay_range = 180  # set to 180 or 90 depending on 1/2 or 1/4 wavelength respectively
tracking_window = 100  # how much you want to incorporate a moving average

'''Setup'''
samp_rate = 30e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
rx_lo = 915e6 #2.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
rx_gain1 = 40
tx_lo = rx_lo
tx_gain = -3
fc0 = int(200e3)

''' Set distance between Rx antennas '''
wavelength = 3E8/rx_lo              # wavelength of the RF carrier
d = d_wavelength*wavelength         # distance between elements in meters
print("Set distance between Rx Antennas to ", int(d*1000), "mm")

'''Create Radios'''
sdr = adi.ad9361(uri='ip:192.168.2.1')

'''Configure properties for the Rx Pluto'''
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(fc0*3)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = int(NumSamples)
sdr._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
sdr.tx_rf_bandwidth = int(fc0*3)
sdr.tx_lo = int(rx_lo)
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = int(tx_gain)
sdr.tx_hardwaregain_chan1 = int(-88)
sdr.tx_buffer_size = int(2**18)

'''Program Tx and Send Data'''
fs = int(sdr.sample_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr.tx([iq0,iq0])  # Send Tx data.

# Assign frequency bins and "zoom in" to the fc0 signal on those frequency bins
xf = np.fft.fftfreq(NumSamples, ts)
xf = np.fft.fftshift(xf)/1e6
signal_start = int(NumSamples*(samp_rate/2+fc0/2)/samp_rate)
signal_end = int(NumSamples*(samp_rate/2+fc0*2)/samp_rate)

def calcTheta(phase):
    # calculates the steering angle for a given phase delta (phase is in deg)
    # steering angle is theta = arcsin(c*deltaphase/(2*pi*f*d)
    arcsin_arg = np.deg2rad(phase)*3E8/(2*np.pi*rx_lo*d)
    arcsin_arg = max(min(1, arcsin_arg), -1)     # arcsin argument must be between 1 and -1, or numpy will throw a warning
    calc_theta = np.rad2deg(np.arcsin(arcsin_arg))
    return calc_theta

def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_shift, s_dbfs

def monopulse_angle(array1, array2):
    ''' Correlate the sum and delta signals  '''
    # Since our signals are closely aligned in time, we can just return the 'valid' case where the signals completley overlap
    # We can do correlation in the time domain (probably faster) or the freq domain
    # In the time domain, it would just be this:
    # sum_delta_correlation = np.correlate(delayed_sum, delayed_delta, 'valid')
    # But I like the freq domain, because then I can focus just on the fc0 signal of interest
    sum_delta_correlation = np.correlate(array1[signal_start:signal_end], array2[signal_start:signal_end], 'valid')
    angle_diff = np.angle(sum_delta_correlation)
    return angle_diff

def scan_for_DOA():
    # go through all the possible phase shifts and find the peak, that will be the DOA (direction of arrival) aka steer_angle
    data = sdr.rx()
    Rx_0=data[0]
    Rx_1=data[1]
    peak_sum = []
    peak_delta = []
    monopulse_phase = []
    if DEBUG:
        plot_generic_signal("Rx_0", Rx_0, sdr.sample_rate)
    delay_phases = np.arange(-phase_delay_range, phase_delay_range, 2)    # phase delay in degrees
    for phase_delay in delay_phases:   
        delayed_Rx_1 = Rx_1 * np.exp(1j*np.deg2rad(phase_delay+phase_cal))
        delayed_sum = Rx_0 + delayed_Rx_1
        delayed_delta = Rx_0 - delayed_Rx_1
        delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
        delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
        mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)
        
        peak_sum.append(np.max(delayed_sum_dbfs))
        peak_delta.append(np.max(delayed_delta_dbfs))
        monopulse_phase.append(np.sign(mono_angle))
        
    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum==peak_dbfs)
    peak_delay = delay_phases[peak_delay_index[0][0]]
    steer_angle = int(calcTheta(peak_delay))

    print("Peak at:\t" + str(peak_delay + phase_cal) + " (degree phase shift)")
    print("Phase cal:\t" + str(phase_cal) + " (degree phase shift)")
    print("Adjusted peak at:\t" + str(peak_delay) + " (degree phase shift)")
    print("Steering angle:\t" + str(steer_angle) + " (degree phase shift)")
    
    return delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase

def Tracking(last_delay):
    # last delay is the peak_delay (in deg) from the last buffer of data collected
    data = sdr.rx()
    Rx_0=data[0]
    Rx_1=data[1]
    delayed_Rx_1 = Rx_1 * np.exp(1j*np.deg2rad(last_delay+phase_cal))
    delayed_sum = Rx_0 + delayed_Rx_1
    delayed_delta = Rx_0 - delayed_Rx_1
    delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
    delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
    mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)
    phase_step= 1
    if np.sign(mono_angle) > 0:
        new_delay = last_delay - phase_step
    else:
        new_delay = last_delay + phase_step
    return new_delay

def calibrate_phase_delay():
    delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase = scan_for_DOA()
    return peak_delay

def plot_generic_signal(title, signal, sample_rate):
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


'''Collect Data'''
for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations
    data = sdr.rx()
    
#scan once to get the direction of arrival (steer_angle) as the initial point for our monopulse tracker
delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase = scan_for_DOA()
delay = peak_delay  # this will be the starting point if we are doing monopulse tracking
delay = Tracking(delay)
first_theta = calcTheta(delay)
tracking_angles = np.ones(tracking_window) * first_theta

'''Setup Polar Plot Window'''
# Setup the plot window
win = pg.GraphicsLayoutWidget(show=True)
p1 = win.addPlot()
p1.setAspectLocked()
p1.hideAxis('bottom')
p1.hideAxis('left')
p1.setXRange(-1, 1)
p1.setYRange(-1, 1)
p1.setTitle('Monopulse Tracking: Compass View', **{'color': '#FFF', 'size': '14pt'})

# Circle for compass boundary
circle = QtWidgets.QGraphicsEllipseItem(-1, -1, 2, 2)
circle.setPen(pg.mkPen('w'))
p1.addItem(circle)

# Line to indicate the steering angle
line = pg.PlotDataItem()
p1.addItem(line)

# Sample function to simulate the tracking angle update
def update_compass():
    global tracking_angles, phase_cal
    delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase = scan_for_DOA()
    delay = peak_delay  # this will be the starting point if we are doing monopulse tracking
    delay = Tracking(delay)
    tracking_angles = np.append(tracking_angles, calcTheta(delay))
    tracking_angles = tracking_angles[1:]
    print(f"Window averaged tracking angle: {np.mean(tracking_angles[-1])}")  # Print the current tracking angle
    print(f"Phase cal: " + str(phase_cal))  # Print the current tracking angle

    disp_steer_angle = np.mean(tracking_angles[-1]) + 90 # +90 to treat vertical line as 0 degrees
    disp_steer_angle_rad = np.deg2rad(disp_steer_angle)  # Convert latest angle to radians and shift for viewing purposes

    x = [0, np.cos(disp_steer_angle_rad)]
    y = [0, np.sin(disp_steer_angle_rad)]
    line.setData(x, y)

# Function to be called by the button
def on_button_click():
    global phase_cal
    phase_cal = phase_cal + calibrate_phase_delay()  # this is a weird way of doing this but it works
    update_compass()

# Timer to update the plot
timer = QtCore.QTimer()
timer.timeout.connect(update_compass)
timer.start(100)

if __name__ == '__main__':
    app = pg.mkQApp()

    # Create a button
    button = QtWidgets.QPushButton('Update Compass')
    button.clicked.connect(on_button_click)

    # Add the button to the window layout
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(button)
    
    # Create a widget to contain the plot and button, and set the layout
    container = QtWidgets.QWidget()
    container.setLayout(layout)
    
    # Create a layout for the main window
    main_layout = QtWidgets.QHBoxLayout()
    main_layout.addWidget(win)
    main_layout.addWidget(container)
    
    # Set up the window and show
    window = QtWidgets.QWidget()
    window.setLayout(main_layout)
    window.show()

    if app.instance() is not None:
        app.instance().exec()

sdr.tx_destroy_buffer()