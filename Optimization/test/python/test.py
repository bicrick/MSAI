import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BlockCollisionSimulator:
    def __init__(self, m1=1, m2=100, v1=0, v2=-1, wall_position=0):
        # Initialize masses and velocities
        self.m1 = m1  
        self.m2 = m2  
        self.v1 = v1  
        self.v2 = v2  
        self.wall = wall_position
        
        # Position of blocks (start with block 1 near wall, block 2 to the right)
        self.x1 = wall_position + 0.5  # Add offset to prevent immediate wall collision
        self.x2 = 4
        
        self.collision_count = 0
        self.collision_times = []
        self.history = {'x1': [], 'x2': [], 't': []}

    def simulate(self, t_max=10, dt=0.00001):  # Much smaller dt for precision
        t = 0
        while t < t_max and self.collision_count < 10000:
            # Store current state
            if len(self.history['t']) % 100 == 0:  # Store every 100th state to save memory
                self.history['x1'].append(self.x1)
                self.history['x2'].append(self.x2)
                self.history['t'].append(t)
            
            # Calculate next positions
            next_x1 = self.x1 + self.v1 * dt
            next_x2 = self.x2 + self.v2 * dt
            
            # Check for wall collision
            if next_x1 <= self.wall:
                self.v1 = -self.v1
                self.collision_count += 1
                self.collision_times.append(t)
            
            # Check for block collision (assuming blocks of unit width)
            elif next_x1 + 1 >= next_x2:  # Use next positions for collision check
                # Elastic collision formulas
                v1_new = ((self.m1 - self.m2) * self.v1 + 2 * self.m2 * self.v2) / (self.m1 + self.m2)
                v2_new = ((self.m2 - self.m1) * self.v2 + 2 * self.m1 * self.v1) / (self.m1 + self.m2)
                self.v1, self.v2 = v1_new, v2_new
                self.collision_count += 1
                self.collision_times.append(t)
            
            # Update positions
            self.x1 = next_x1
            self.x2 = next_x2
            
            t += dt
            
            # Stop if blocks are moving away from each other and no more collisions possible
            if self.v1 < self.v2 and self.x1 + 1 < self.x2:
                break

        return self.collision_count

# Run simulation
sim = BlockCollisionSimulator(m1=1, m2=100)
collisions = sim.simulate()
print(f"Number of collisions: {collisions}")

# Create animation
fig, ax = plt.subplots(figsize=(10, 5))
block1, = ax.plot([], [], 's', markersize=20, color='gray')
block2, = ax.plot([], [], 's', markersize=20, color='lightblue')

def init():
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 0.5)
    return block1, block2

def animate(i):
    frame = i * 10  # Skip frames for smoother animation
    if frame < len(sim.history['t']):
        block1.set_data([sim.history['x1'][frame]], [0])
        block2.set_data([sim.history['x2'][frame]], [0])
    return block1, block2

anim = FuncAnimation(fig, animate, init_func=init, frames=len(sim.history['t'])//10,
                    interval=20, blit=True)
plt.show()
