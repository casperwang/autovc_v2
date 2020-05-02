#import the pyplot and wavfile modules 

import matplotlib.pyplot as plot

from scipy.io import wavfile

 

# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read('./result_wav/results_iter4999_KK1.wav')

 

# Plot the signal read from wav file

plt1 = plot.subplot(321)


 

plot.plot(signalData)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

 

plt2 = plot.subplot(322)

plot.specgram(signalData,Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')

 
samplingFrequency, signalData = wavfile.read('./result_wav/results_iter8999_KK1.wav')

 

# Plot the signal read from wav file

plt3 = plot.subplot(323, sharex = plt1)


 

plot.plot(signalData)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

 

plt4 = plot.subplot(324, sharex = plt2)

plot.specgram(signalData,Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')


samplingFrequency, signalData = wavfile.read('./data/VCTK/p225/p225_001.wav')

 

# Plot the signal read from wav file

plt3 = plot.subplot(325, sharex = plt1)


 

plot.plot(signalData)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

 

plt4 = plot.subplot(326, sharex = plt2)

plot.specgram(signalData,Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')


plot.show()