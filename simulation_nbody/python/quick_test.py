import rebound
import numpy as np
sim = rebound.Simulation()
sim.add(m=1.0)
for i in range(0, 1000):
    sim.add(m=1.0e-3, a=np.random.uniform(low=0.8, high=1.2))
print 'Starting Simulation'
sim.integrate(100.)
sim.status()
