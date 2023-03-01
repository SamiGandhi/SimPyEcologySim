import csv
import json

from SIMULATION_PROCESSES import ahcen_method
from SIMULATION_PROCESSES import ameliorated_method_knn
from SIMULATION_PROCESSES import ameliorated_method_mog2
from SIMULATION_PROCESSES import no_region_of_interest
from COLORS import ConsoleColor
import cv2 as cv


class SensingNode:
    def __init__(self, env, name, video_file, roi_method, communication_channel, color):
        self.env = env
        self.name = name
        self.communication_chanel = communication_channel
        self.video_file = video_file
        self.roi_method = roi_method
        self.color = color
        self.roi_size_data = []
        self.frame_size_data = []


    def run(self):
        cap = cv.VideoCapture(self.video_file)
        framess = 0
        while True:
            ret, frame = cap.read()
            captured_time = self.env.now
            print(self.color+f"Node {self.name} captured a frame at : {captured_time}"+ConsoleColor.RESET)
            if ret:
                raw_size = frame.size * frame.itemsize
                frame_data = raw_size, captured_time
                self.frame_size_data.append(frame_data)
                # Extract the Region of Interest (ROI) from the frame using the proper method
                if self.roi_method == "NO ROI":
                    roi = no_region_of_interest(frame)
                elif self.roi_method == "AHCEN METHOD":
                    # When using Ahcen's method we must use two frames frame_n and frame_n_1
                    frame_n_1 = frame
                    ret, frame_n = cap.read()
                    roi = ahcen_method(frame_n, frame_n_1)
                elif self.roi_method == "KNN":
                    roi = ameliorated_method_knn(frame)
                elif self.roi_method == "MOG2":
                    roi = ameliorated_method_mog2(frame)

                # Send the ROI to the cluster head if detected else do nothing
                if roi is not None:
                    roi_extraction_time = self.env.now
                    print(self.color+f"Node {self.name} extracts the roi at time : {roi_extraction_time}"+ConsoleColor.RESET)
                    # Send the frame time stamped with the id of the sensing node the time stamp and the extracted roi
                    send_roi = (self.name, captured_time, roi)
                    yield self.env.timeout(1)
                    # Send data to other nodes
                    print(self.color+f"Node {self.name} sending the captured roi at {roi_extraction_time} at time : {self.env.now}"+ConsoleColor.RESET)
                    self.communication_chanel.put(send_roi)
                    roi_size = roi.size * roi.itemsize
                    print("frame size ===> "+str(raw_size))
                    print("roi size ===> " + str(roi_size))
                    roi_data = roi_size, roi_extraction_time, self.env.now
                    self.roi_size_data.append(roi_data)

                else:
                    yield self.env.timeout(1)
                    continue
            else:
                print(str(self.env.now))
                break
            # write the data in a file

        with open(str(self.name) + "_" + str(self.roi_method) + '_roi_size_data.json', 'w') as f:
            json.dump(self.roi_size_data, f)
        with open(str(self.name) + "_" + str(self.roi_method) + '_frame_size_data.json', 'w') as f:
            json.dump(self.roi_size_data, f)
        print("Simulation ended and files writen.")
