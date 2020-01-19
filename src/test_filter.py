
import numpy as np
import matplotlib.pyplot as plt

value = 600
ini_value = 0

PREV_FILTER = 0.95
NEXT_FILTER = 0.05
current_value = []
current_value.append(ini_value)

time = []
time.append(0.0)
for k in range(1,125):
    time.append(k)
    current_value.append(PREV_FILTER*current_value[k-1]+NEXT_FILTER*value)
    print(current_value[k])
plt.plot(time, current_value)
plt.show()