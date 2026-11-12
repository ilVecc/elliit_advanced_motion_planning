# OPTIMIZATION-BASED CONTROL
# 
# Marcucci, T., Nobel, P., Tedrake, R., & Boyd, S. (2024). 
#   "Fast path planning through large collections of safe boxes". 
#   IEEE Transactions on Robotics, 40, 3795-3811.
# DOI : https://doi.org/10.1109/TRO.2024.3434168

import numpy as np
import fastpathplanning as fpp
import matplotlib.pyplot as plt

L = np.array([
    [-9, 1],
    [-13, 5],
    [-15, 5],
    [-12, 7],
])
U = np.array([
    [-6, 6],
    [-7, 6],
    [-12, 9],
    [-7, 9],
])
S = fpp.SafeSet(L, U)


T = 30
p_init = np.array([-8, 2])
p_term = np.array([-8.0, 8.0])
alpha = [1, 10, 10]
p = fpp.plan(S, p_init, p_term, T, alpha)

# t = 5.
# dp = p.derivative()
# ddp = dp.derivative()

# p_t = p(t)
# dp_t = dp(t)
# ddp_t = ddp(t)

plt.figure()
S.plot2d()
p.plot2d()
plt.gca().set_aspect("equal")
plt.show()
