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
INCREMENTAL_TRACKING = False  # original method by John Kraft to compare phase to last value. Probably faster but less accurate

'''User inputs'''
phase_cal = 0  # change this based on calibration to the phase shift value when AoA = 0
d_wavelength = 0.50  # distance between elements as a fraction of wavelength.  This is normally 0.5
phase_delay_range = 180  # set to 180 or 90 depending on 1/2 or 1/4 wavelength respectively
phase_cal_window_size = 10
phase_window_size = 5
angle_window_size = 6

'''Setup'''
samp_rate = 30e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**10
rx_lo = 2.3e9
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
#sdr.tx([iq0,iq0])  # Send Tx data.

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
    calc_theta = -1 * calc_theta
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
    ''' Correlate the sum and delta signals'''
    # Since our signals are closely aligned in time, we can just return the 'valid' case where the signals completley overlap
    # We can do correlation in the time domain (probably faster) or the freq domain
    # In the time domain, it would just be this:
    # sum_delta_correlation = np.correlate(delayed_sum, delayed_delta, 'valid')
    # But I like the freq domain, because then I can focus just on the fc0 signal of interest
    sum_delta_correlation = np.correlate(array1[signal_start:signal_end], array2[signal_start:signal_end], 'valid')
    angle_diff = np.angle(sum_delta_correlation)
    return angle_diff

def multi_DOA_scan(phase_cal_input, iters):
    peak_delays = np.zeros(iters)
    for i in range(iters):
        delay_phases, peak_dbfs, peak_delay, peak_sum = scan_for_DOA(phase_cal_input=phase_cal_input)
        peak_delays[i] = peak_delay
    phase_delay = np.mean(remove_outliers(peak_delays))
    return phase_delay

def scan_for_DOA(phase_cal_input):
    # go through all the possible phase shifts and find the peak, that will be the DOA (direction of arrival) aka steer_angle
    data = sdr.rx()
    Rx_0=data[0]
    Rx_1=data[1]
    peak_sum = []
    peak_delta = []
    monopulse_phase = []
    if DEBUG:
        #plot_signals("Rx_0", Rx_0, sdr.sample_rate)
        plot_signals(sdr.sample_rate, Rx_0=Rx_0, Rx_1=Rx_1)
        plot_and_estimate_phase(Rx_0, Rx_1, sdr.sample_rate, "Rx_0", "Rx_1")
    delay_phases = np.arange(-phase_delay_range, phase_delay_range, 2)    # phase delay in degrees
    for phase_delay in delay_phases:   
        delayed_Rx_1 = Rx_1 * np.exp(1j*np.deg2rad(phase_delay+phase_cal_input))  # (180 + phase_delay - phase_cal) % 360 + 180
        delayed_sum = Rx_0 + delayed_Rx_1
        delayed_delta = Rx_0 - delayed_Rx_1
        delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
        delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
        #mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)
        
        peak_sum.append(np.max(delayed_sum_dbfs))
        #peak_delta.append(np.max(delayed_delta_dbfs))
        #monopulse_phase.append(np.sign(mono_angle))
        
    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum==peak_dbfs)
    peak_delay = delay_phases[peak_delay_index[0][0]]
    #steer_angle = int(calcTheta(peak_delay))

    # if DEBUG:
    #     print("Peak at:\t" + str(peak_delay + phase_cal) + " (degree phase shift)")
    #     print("Phase cal:\t" + str(phase_cal) + " (degree phase shift)")
    #     print("Adjusted peak at:\t" + str(peak_delay) + " (degree phase shift)")
    #     print("Steering angle:\t" + str(steer_angle) + " (degree phase shift)")
    
    return delay_phases, peak_dbfs, peak_delay, peak_sum #, peak_delta, monopulse_phase

def Tracking(last_delay):
    # HOW THIS WORKS
    # Reruns the DoA scan but only for the calibrated delay. Then adjusts the delay by 1 at a time based on how the reference angle changes
    # last delay is the peak_delay (in deg) from the last buffer of data collected
    # NOT CURRENTLY IN USE
    data = sdr.rx()
    Rx_0=data[0]
    Rx_1=data[1]
    delayed_Rx_1 = Rx_1 * np.exp(1j*np.deg2rad(last_delay+phase_cal))
    delayed_sum = Rx_0 + delayed_Rx_1
    delayed_delta = Rx_0 - delayed_Rx_1
    delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
    delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
    mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)
    phase_step= 5  # TODO experiment with slightly larger value to see if it moves faster
    if np.sign(mono_angle) > 0:
        new_delay = last_delay - phase_step
    else:
        new_delay = last_delay + phase_step
    return new_delay

def remove_outliers(arr, threshold=2):
    """
    Removes outliers from a NumPy array using the IQR method.
    
    Parameters:
        arr (numpy.ndarray): Input array containing only numbers.
        threshold (float): Multiplier for the IQR to determine outlier boundaries (default is 1.5).
    
    Returns:
        numpy.ndarray: Array with outliers removed.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    
    # Compute Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    
    # Compute IQR
    IQR = Q3 - Q1
    
    # Determine bounds for non-outliers
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Filter array to remove outliers
    return arr[(arr >= lower_bound) & (arr <= upper_bound)]

def plot_signals(sample_rate, **signals):
    num_signals = len(signals)
    rows = num_signals  # Each signal gets a pair of subplots (time + frequency)

    plt.figure(figsize=(12, 4 * num_signals))

    for i, (title, signal) in enumerate(signals.items()):
        t = np.linspace(0, (len(signal) - 1) / sample_rate, len(signal))

        # Time domain plot
        plt.subplot(rows, 2, 2 * i + 1)
        plt.plot(t, signal, label=f"{title} - Time Domain", color=np.random.rand(3,))
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Time Domain")
        plt.legend()
        plt.grid()

        # Frequency domain plot
        freq = np.fft.fftfreq(len(signal), d=1/sample_rate)
        fft_signal = np.fft.fft(signal)
        plt.subplot(rows, 2, 2 * i + 2)
        plt.plot(freq[:len(freq)//2], np.abs(fft_signal[:len(freq)//2]), label=f"{title} - Frequency Domain", color=np.random.rand(3,))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"{title} - Frequency Domain")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

def plot_and_estimate_phase(signal1, signal2, sample_rate, title1="Signal 1", title2="Signal 2"):
    t = np.linspace(0, (len(signal1) - 1) / sample_rate, len(signal1))

    # Plot the two signals
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal1, label=title1, color='b', linestyle='-')
    plt.plot(t, signal2, label=title2, color='r', linestyle='--', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Overlapping Signals")
    plt.legend()
    plt.grid()
    plt.show()

    # Compute phase difference using FFT
    fft1 = np.fft.fft(signal1)
    fft2 = np.fft.fft(signal2)

    # Find dominant frequency
    freq = np.fft.fftfreq(len(signal1), d=1/sample_rate)
    idx = np.argmax(np.abs(fft1))  # Index of dominant frequency

    # Compute phase difference at dominant frequency
    phase_diff = np.angle(fft2[idx]) - np.angle(fft1[idx])
    phase_diff_deg = np.degrees(phase_diff)  # Convert to degrees

    print(f"Estimated Phase Difference: {phase_diff_deg:.2f} degrees")

    return phase_diff_deg

def plot_values(data, title="Plot", xlabel="Index", ylabel="Value", line_style='-o'):
    """
    Plots values from a NumPy array or a list using matplotlib.

    Parameters:
        data (list or np.ndarray): List or NumPy array of numerical values to plot.
        title (str): Title of the plot (default: "Plot").
        xlabel (str): Label for the x-axis (default: "Index").
        ylabel (str): Label for the y-axis (default: "Value").
        line_style (str): Line style for the plot (default: '-o').
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("Input data must be a list or a NumPy array of numerical values.")

    # Convert list to NumPy array if necessary
    data = np.asarray(data)

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input data must contain numerical values.")

    plt.figure(figsize=(8, 5))
    plt.plot(data, line_style, markersize=5, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Global flag for pause state
paused = False

def toggle_pause():
    global paused, pauseButton
    paused = not paused
    if paused:
        pauseButton.setText("Play")
    else:
        pauseButton.setText("Pause")


'''Collect Data'''
for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations
    data = sdr.rx()
    
#scan once to get the direction of arrival (steer_angle) as the initial point for our monopulse tracker
phase_cal = multi_DOA_scan(0, phase_cal_window_size)
delay_phases, peak_dbfs, peak_delay, peak_sum = scan_for_DOA(phase_cal)

if INCREMENTAL_TRACKING:
    delay = peak_delay  # this will be the starting point if we are doing monopulse tracking
    delay = Tracking(delay)

first_theta = calcTheta(peak_delay)
tracking_angles = np.ones(angle_window_size) * first_theta

'''Setup Polar Plot Window'''
# Setup the plot window
win = pg.GraphicsLayoutWidget(show=True)
p1 = win.addPlot()
p1.setAspectLocked()
p1.hideAxis('bottom')
p1.hideAxis('left')
p1.setXRange(-1, 1)
p1.setYRange(-1, 1)
p1.setTitle('Signal Tracking Compass', **{'color': '#FFF', 'size': '14pt'})

# Circle for compass boundary
circle = QtWidgets.QGraphicsEllipseItem(-1, -1, 2, 2)
circle.setPen(pg.mkPen('w'))
p1.addItem(circle)

# Line to indicate the steering angle
line = pg.PlotDataItem(pen=pg.mkPen(width=10))
p1.addItem(line)

# NEW: Create a TextItem for the steering angle.
# This will be placed on top of the compass circle (adjust position as needed).
steeringAngleText = pg.TextItem(text="Steering Angle: 0°", color='w', anchor=(0.5,0))
# Place the text at (0, -0.1) so that it is just below the line pointing to 0°.
steeringAngleText.setPos(0, -0.3)
p1.addItem(steeringAngleText)
font = QtGui.QFont("Arial", 32,)
steeringAngleText.setFont(font)

'''Time domain plots'''
# Create a new GraphicsLayoutWidget for time-domain plots
timePlotWidget = pg.GraphicsLayoutWidget()
# Create three PlotWidgets for Rx0, Rx1, and Rx1 shifted.
rx0_rx1_Plot = timePlotWidget.addPlot(title="Rx0 and Rx1 Signals")
rx0_rx1_Plot.showGrid(x=True, y=True)
rx0_rx1_Plot.addLegend()
timePlotWidget.nextRow()
rx1ShiftPlot = timePlotWidget.addPlot(title="Rx1 Shifted Signal (Real)")
rx1ShiftPlot.showGrid(x=True, y=True)
rx1ShiftPlot.addLegend()

# We keep the existing compass widget "win" (which contains p1)
# and add the timePlotWidget as a second widget in the main layout.
main_layout = QtWidgets.QHBoxLayout()
main_layout.addWidget(win)          # compass view
main_layout.addWidget(timePlotWidget) # time-domain plots

# Function to simulate the tracking angle update
def update_compass():
    global tracking_angles, phase_cal, delay

    # Check if paused; if yes, skip updating.
    if paused:
        return
    
    if INCREMENTAL_TRACKING:
        delay = Tracking(delay)  # TODO: test not using this
    else:
        peak_delay = multi_DOA_scan(phase_cal, phase_window_size)
        delay = peak_delay

    tracking_angles = np.append(tracking_angles, calcTheta(delay))
    tracking_angles = tracking_angles[1:]  # remove oldest measurement
    tracking_angles_inliers = remove_outliers(tracking_angles)
    aoa = np.mean(tracking_angles_inliers)

    if DEBUG:
        print(f"Tracking angles:\n{tracking_angles}")
        print(f"Inlier tracking angles:\n{tracking_angles_inliers}")
        print(f"Window averaged tracking angle:\n{aoa}")
        #print(f"Window averaged tracking angle: {aoa}")  # Print the current tracking angle
        #print(f"Phase cal: " + str(phase_cal))  # Print the current tracking angle

    disp_aoa = aoa + 90 # +90 to treat vertical line as 0 degrees
    disp_aoa_rad = np.deg2rad(disp_aoa)  # Convert latest angle to radians and shift for viewing purposes

    x = [0, np.cos(disp_aoa_rad)]
    y = [0, np.sin(disp_aoa_rad)]
    line.setData(x, y)

 #  phase_cal_label.setText(f"Phase Calibration:")
    phase_cal_num.setText(f"{phase_cal:.2f}°")
 #   phase_delay_label.setText(f"Average Phase Delay:")
    phase_delay_num.setText(f"{delay:.2f}°")
    steeringAngleText.setText(f"AOA: {aoa:.2f}°")

    # --- Update the time-domain plots using pyqtgraph ---
    # Acquire new block of Rx data
    data = sdr.rx()
    Rx0 = data[0]
    Rx1 = data[1]
    # Apply the current phase shift (use peak_delay as phase shift value, for example)
    Rx1_shifted = Rx1 * np.exp(1j * np.deg2rad(delay+phase_cal)) ##delay - phase cal gives  inverted signal, delay + phase cal seems to work
    RX1_calibrated = Rx1 * np.exp(1j * np.deg2rad(phase_cal))
    t = np.arange(len(Rx0)) / samp_rate

    # Update plots: setData expects x and y arrays.
    rx0_rx1_Plot.plot(t, np.real(Rx0), clear=True, pen='cyan', name="Rx0")
    rx0_rx1_Plot.plot(t, np.real(RX1_calibrated), clear=False, pen='orange', name="Rx1")
    rx1ShiftPlot.plot(t, np.real(Rx0), clear=True, pen='cyan', name="Rx0")
    rx1ShiftPlot.plot(t, np.real(Rx1_shifted), clear=False, pen='g', name="Rx1 Shifted")

# Function to be called by the button
def phase_cal_button_click():
    global phase_cal, phase_window_size
    phase_cal = multi_DOA_scan(0, phase_cal_window_size)
    update_compass()

# Timer to update the plot
timer = QtCore.QTimer()
timer.timeout.connect(update_compass)
timer.start(100)

if __name__ == '__main__':
    app = pg.mkQApp()

    # Create a button
    button = QtWidgets.QPushButton('Phase Calibration')
    button.clicked.connect(phase_cal_button_click)

    # Create the pause/play button
    pauseButton = QtWidgets.QPushButton("Pause")
    pauseButton.clicked.connect(toggle_pause)
    
    phase_cal_label = QtWidgets.QLabel("Phase Calibration:")  # Initialize label
    phase_cal_label.setStyleSheet("font-size: 14pt;")
    phase_cal_num = QtWidgets.QLabel("0.00°")
    phase_cal_num.setStyleSheet("font-size: 32pt;")
    phase_delay_label = QtWidgets.QLabel("Average Phase Delay:")  # Initialize label
    phase_delay_label.setStyleSheet("font-size: 14pt;")
    phase_delay_num = QtWidgets.QLabel("0.00°")
    phase_delay_num.setStyleSheet("font-size: 32pt;")
    #steer_angle_label = QtWidgets.QLabel("Steering Angle: 0.00°")  # Initialize label
    
    # Add the elements to the window layout
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(button)
    layout.addWidget(pauseButton)      # add the pause/play button
    layout.addWidget(phase_cal_label)
    layout.addWidget(phase_cal_num)
    layout.addWidget(phase_delay_label)
    layout.addWidget(phase_delay_num)
    #layout.addWidget(steer_angle_label)
    
    # Create a widget to contain the plot and button, and set the layout
    container = QtWidgets.QWidget()
    container.setLayout(layout)
    
    # Create a layout for the main window that includes:
# 1. The compass widget ("win")
# 2. The time-domain plot widget ("timePlotWidget")
# 3. The container with your buttons/labels ("container")
main_layout = QtWidgets.QHBoxLayout()
main_layout.addWidget(win)           # Compass view
main_layout.addWidget(timePlotWidget)  # Time-domain plots
main_layout.addWidget(container)       # Control panel

# Create the main window, set the layout, and show it.
window = QtWidgets.QWidget()
window.setLayout(main_layout)
window.show()

if app.instance() is not None:
    app.instance().exec()

sdr.tx_destroy_buffer()