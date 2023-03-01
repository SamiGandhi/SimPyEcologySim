import simpy
import wsnsimpy.wsnsimpy
import COLORS
from CLUSTER_HEAD import ClusterHead
from SENSING_NODE import SensingNode


class Environment:

    def __init__(self, name):
        self.name = name
        self.environment = simpy.Environment()
        self.clusters = []
        self.sensing_nodes = []
        self.channels = []

    # Cluster head and the sensing node share the same communication channel
    def add_cluster_head(self, cluster_head):
        self.clusters.append(cluster_head)

    def add_sensing_node(self, sensing_node):
        self.sensing_nodes.append(sensing_node)

    def add_communication_chanel(self):
        self.channels.append(simpy.Store(self.environment, capacity=4))

    def start_simulation(self):
        for sensing_node in self.sensing_nodes:
            self.environment.process(sensing_node.run())

        for cluster_head in self.clusters:
            self.environment.process(cluster_head.run())

        self.environment.run()


if __name__ == "__main__":
    env = Environment("catzone")
    env.add_communication_chanel()
    cluster_head = ClusterHead(env.environment, "SIFT", "BRUTE FORCE", env.channels[0], COLORS.ConsoleColor.YELLOW)
    # Replace the file path with the video file
    sensing_node1 = SensingNode(env.environment, "SENS1", "file path", "KNN", env.channels[0], COLORS.ConsoleColor.GREEN)
    sensing_node2 = SensingNode(env.environment, "SENS2", "file path", "KNN", env.channels[0], COLORS.ConsoleColor.PURPLE)
    env.add_cluster_head(cluster_head)
    env.add_sensing_node(sensing_node2)
    env.add_sensing_node(sensing_node1)
    env.start_simulation()
