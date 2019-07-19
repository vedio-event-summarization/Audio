import math
import wave
import pyaudio
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import sys
import time

global k

CHUNK = 1024
RATE = 24000
plot = 20

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,input_device_index=2,
                  frames_per_buffer=CHUNK)
                  
def goertzel(samples, sample_rate, *freqs):
    
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

class Plot2D():
    def __init__(self):
        self.traces = dict()

        #QtGui.QApplication.setGraphicsSystem('raster')
        self.app = QtGui.QApplication([])
        #mw = QtGui.QMainWindow()
        #mw.resize(800,800)

        self.win = pg.GraphicsWindow(title="SDR")
        self.win.resize(1000,600)
        self.win.setWindowTitle('Dinusha')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.canvas = self.win.addPlot(title=" Goertzel filter ")

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def trace(self,name,dataset_x,dataset_y):
        if name in self.traces:
            self.traces[name].setData(dataset_x,dataset_y)
        else:
            self.traces[name] = self.canvas.plot(pen='g')
            
e1 = []#witsil blown time array


if __name__ == '__main__':
    p = Plot2D()
    x = np.arange(plot)
    
    start_t = time.time()
    
    def update():
            k = []
            for i in range(0,plot):
        	data = np.fromstring(stream.read(CHUNK,exception_on_overflow=False),dtype=np.int16)
        	freqs, results = goertzel(data, 24000, (3900, 4100))
        	#print(freqs)
        	#print(results[1][2])
        	k.append(results[3][2])
        	if results[3][2] > 5e9:
        	   t = time.time() - start_t
        	   e1.append(t)
        	   print('4k tone @ :',t)
        	
            p.trace("power",x,k)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(5)

    p.start()
    
    
    
    
    

















