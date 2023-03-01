import simpy
import random

# Parameters
NUM_PACKETS = 100
PACKET_SIZE = 1000  # bytes
BITRATE = 1000  # bits per second
DISTANCE = 100  # meters
ATTENUATION_FACTOR = 0.1
COLLISION_PROBABILITY = 0.1
DATA_LOSS_PROBABILITY = 0.1

# Simulate a wireless channel
def wireless_channel(env, packet):
    # Calculate the signal strength based on distance and attenuation factor
    signal_strength = 1 / (DISTANCE ** ATTENUATION_FACTOR)
    # Calculate the probability of packet loss based on signal strength
    packet_loss_probability = 1 - signal_strength
    # Calculate the probability of collision
    collision_probability = COLLISION_PROBABILITY
    # Check if the packet is lost
    if random.random() < packet_loss_probability:
        print(f"Packet {packet} lost")
        return
    # Check if there is a collision
    if random.random() < collision_probability:
        print(f"Packet {packet} collided")
        # Wait for a random backoff time before retrying
        yield env.timeout(random.uniform(0, 1/BITRATE))
        # Retry sending the packet
        yield env.process(wireless_channel(env, packet))
    else:
        # Transmit the packet
        print(f"Packet {packet} transmitted")
        yield env.timeout(PACKET_SIZE/BITRATE)
        # Check if the packet is lost during transmission
        if random.random() < DATA_LOSS_PROBABILITY:
            print(f"Packet {packet} lost during transmission")
        else:
            print(f"Packet {packet} received")

# Simulate the wireless channel for a number of packets
def run_simulation(env):
    for i in range(NUM_PACKETS):
        env.process(wireless_channel(env, i))
        # Wait for a random inter-packet interval
        yield env.timeout(random.expovariate(1/BITRATE))

# Run the simulation
env = simpy.Environment()
env.process(run_simulation(env))
env.run()
