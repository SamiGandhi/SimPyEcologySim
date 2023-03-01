import json

from SIMULATION_PROCESSES import sift
from SIMULATION_PROCESSES import orb
from SIMULATION_PROCESSES import kaze
from SIMULATION_PROCESSES import akaze
from SIMULATION_PROCESSES import fast
from SIMULATION_PROCESSES import brute_force_matcher
from SIMULATION_PROCESSES import flann_matcher
from COLORS import ConsoleColor

class ClusterHead:
    def __init__(self, env, feature_extraction_algo, feature_matching_algo, communication_channel, color):
        self.env = env
        self.received_roi = []
        self.feature_extraction_algo = feature_extraction_algo
        self.feature_matching_algo = feature_matching_algo
        self.communication_channel = communication_channel
        self.color = color
        self.latency_data = []
        self.feature_extraction_data = []
        self.feature_matching_data = []
        self.redundant_object_data = []
        self.reception_data = []
        self.sent_data =[]


    def run(self):
        # d = self.communication_channel.get()
        finished_process = 0
        while finished_process < 2:
            yield self.env.timeout(1)
            try:
                 received = self.communication_channel.get()
                 sensing_node, time, roi = received.value
                 print(self.color + f"Cluster head received frame at time {self.env.now}" + " From -> " + str(
                     sensing_node) + " send at :" + str(time) + ConsoleColor.RESET)
                 raw_size = roi.size * roi.itemsize
                 frame_data = raw_size, self.env.now
                 self.reception_data.append(frame_data)
                 latency = self.env.now - time
                 latency_data = latency, self.env.now, sensing_node
                 self.latency_data.append(latency_data)
                 print("Latency:", latency)
                 self.received_roi.append((sensing_node, time, roi))
                 self.process_frames()
            except Exception:
                finished_process += 1


            # after finishing write data

        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_latency_data.json', 'w') as f:
                json.dump(self.latency_data, f)

        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_features_data.json', 'w') as f:
            json.dump(self.feature_extraction_data, f)
        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_features_matching_data.json', 'w') as f:
            json.dump(self.feature_matching_data, f)
        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_redundant_data.json', 'w') as f:
            json.dump(self.redundant_object_data, f)
        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_features_data.json', 'w') as f:
            json.dump(self.feature_extraction_data, f)
        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_features_matching_data.json', 'w') as f:
            json.dump(self.feature_matching_data, f)
        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_reception_data.json', 'w') as f:
            json.dump(self.reception_data, f)
        with open('cluster latency 4'+str(self.feature_extraction_algo)+"_"+str(self.feature_matching_algo)+'_send_data.json', 'w') as f:
            json.dump(self.sent_data, f)
        print("Simulation ended and files writen.")


    def process_frames(self):
            if len(self.received_roi) >= 2:
                self.received_roi.sort(key=lambda x: x[1])
                sensing_node1, time1, roi1 = self.received_roi.pop()
                for sensing_node2, time2, roi2 in self.received_roi:
                    if time2 == time1:
                        sensing_node2, time2, roi2 = self.received_roi.pop()
                        print(self.color + "Ready to extract features using: " + str(
                            self.feature_extraction_algo) + f" from roi captured by {sensing_node1} and {sensing_node2} "
                                                            f"at {time1}" + ConsoleColor.RESET)
                        if self.feature_extraction_algo == "SIFT":
                            kp1, descr1 = sift(roi1)
                            kp2, descr2 = sift(roi2)
                        elif self.feature_extraction_algo == "KAZE":
                            kp1, descr1 = kaze(roi1)
                            kp2, descr2 = kaze(roi2)
                        elif self.feature_extraction_algo == "AKAZE":
                            kp1, descr1 = akaze(roi1)
                            kp2, descr2 = akaze(roi2)
                        elif self.feature_extraction_algo == "ORB":
                            kp1, descr1 = orb(roi1)
                            kp2, descr2 = orb(roi2)
                        elif self.feature_extraction_algo == "FAST":
                            kp1, descr1 = fast(roi1)
                            kp2, descr2 = fast(roi2)
                        feature_extraction_data = len(kp1), len(kp2), self.env.now
                        self.feature_extraction_data.append(feature_extraction_data)
                        print(self.color + "Features Extracted from roi 1 : " + str(len(kp1)) + ConsoleColor.RESET)
                        print(self.color + "Features Extracted from roi 2 : " + str(len(kp2)) + ConsoleColor.RESET)
                        # Match the extracted features using the proper method
                        if self.feature_matching_algo == "BRUTE FORCE":
                            matches = brute_force_matcher(descr1, descr2)
                        elif self.feature_extraction_algo == "FLANN":
                            matches = flann_matcher(descr1, descr2)
                        # Process the matched features
                        self.process_matches(matches,roi1,roi2)

                    else:
                        self.received_roi.append((sensing_node1, time1, roi1))
                        print(self.color + f"roi captured by {sensing_node1} at {time1} \n"
                                           f" && roi captured by {sensing_node1} at {time1}\n"
                                           f" are desynchronized the system cannot perform matching"
                              + ConsoleColor.RESET)
            else:
                print(self.color+"just one region of interests in the waiting queue cannot performa matching"+ConsoleColor.RESET)

    def process_matches(self, matches,roi1, roi2):
        roi1_size = roi1.size * roi1.itemsize
        roi2_size = roi2.size * roi2.itemsize
        if (matches is not None):
            matches_data = len(matches), self.env.now
            self.feature_matching_data.append(matches_data)
            if  len(matches) > 20:
                redundant_object_data = 1, self.env.now
                self.redundant_object_data.append(redundant_object_data)
                print("Redundant object")
                frame_data = roi1_size, self.env.now
                self.sent_data.append(frame_data)
            else:
                print("no redundant object")
                data_size = roi2_size + roi1_size
                frame_data = data_size, self.env.now
                self.sent_data.append(frame_data)
        else:
            matches_data = 0, self.env.now
            self.feature_matching_data.append(matches_data)
            data_size = roi2_size+ roi1_size
            frame_data = data_size, self.env.now
            self.sent_data.append(frame_data)
            print("no redundant object")
        return None