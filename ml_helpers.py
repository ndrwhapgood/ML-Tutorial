#TODO: scale output and test.

import serial
from matplotlib import pyplot as plt
import numpy as np

def get_proximity_data(samples = 1000):
    ser = serial.Serial()
    ser.port = '/dev/ttyACM0'
    ser.baudrate = 19200

    ser.open()
    ser.reset_input_buffer()

    def serial_read_line(obj):
        data = obj.readline()
        return data.decode('utf-8')

    x = range(0,samples)
    # initalize return data y
    y = [0] * samples
    y[1] = 255 # assign this so the graph can scale better.

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x, y)

    for i in range(samples):
        y[i] = int(serial_read_line(ser))

        line1.set_xdata(x)
        line1.set_ydata(y)
    
        # drawing updated values
        figure.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()
        # no need for a delay, the delay happens on the micro controller.
    plt.close(figure)
    return np.asarray(y,dtype=np.float32)

if __name__ == '__main__':
    data = get_proximity_data()
