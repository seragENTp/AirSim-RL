import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import json
from PIL import Image
from pylab import array, arange, uint8 

import cv2

import time
import sys
from scipy import misc
import math

import airsim

class Environment():
    def __init__(self):

       
        # CV params
        self.CV_SLEEP_TIME = 0.05 # 0.05
       
        # both False
       # self.CV_MODE = False
        self.MOVE_RATE = 1.0 #0.2 1.0
        self.HEIGHT = 0
        self.goal = [30.79,-0.09] #cone position
        self.newAirSimClient()

               


    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)

    def newAirSimClient(self):
        client = airsim.MultirotorClient() # 41451
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        current_position = client.simGetVehiclePose().position
        dist = math.sqrt((current_position.x_val - self.goal[0])**2 + (current_position.y_val - self.goal[1])**2)
        self.allLogs  = { 'distance': [dist] }
        self.allLogs = { 'reward': [0] }
        self.client = client

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        #self.INITIAL_YAW = math.atan2(-12.8, -4.9)
        home_pose = airsim.Pose(airsim.Vector3r(0, 0, self.HEIGHT), airsim.utils.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(home_pose, False)
        time.sleep(self.CV_SLEEP_TIME*10.)
        current_position = self.client.simGetVehiclePose().position
        dist = math.sqrt((current_position.x_val - self.goal[0])**2 + (current_position.y_val - self.goal[1])**2)      
        self.allLogs = { 'reward': [0] }
        self.allLogs  = { 'distance': [dist] }

        track = self.goalDirection(self.goal, current_position) 

        state = self.getDepth(track)

        return state
    
    def transformRGB(self, responses, log=False):
        img2d = AirSimClientBase.stringToUint8Array(responses[0].image_data_uint8).reshape(144*2, 256*2, 4)[:, :, :3]
        # img_tensor = img2d.reshape(144*2, 256*2, 3) # IF SIAMESE APPLY TRANSFORMATION TO TARGET IMAGE ALSO!!
        img_tensor = img2d.reshape(1, 144*2, 256*2, 3).transpose(0,3,1,2).astype(np.float32)/ 255.
        return img_tensor

    def transformDepth(self, responses, log=False):
        img2d = AirSimClientBase.stringToUint8Array(responses[0].image_data_uint8).reshape(144*2, 256*2, 1)#[:, :, 3]
        # img_tensor = img2d.reshape(144*2, 256*2, 3) # IF SIAMESE APPLY TRANSFORMATION TO TARGET IMAGE ALSO!!
        img_tensor = img2d.reshape(1, 144*2, 256*2, 1).transpose(0,3,1,2).astype(np.float32)#/ 255.
        return img_tensor

    def getImg(self):
        responses = self.client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
        state = self.transformInput(responses)
        return state

    def getDepth(self,track):
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        
        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        
        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        newImage1 = (maxIntensity)*(image/maxIntensity)**factor
        newImage1 = array(newImage1,dtype=uint8)
        img_dim = newImage1.shape
        #print (img_dim)
        #sys.stdout.flush()
        newImage1 = newImage1.reshape( img_dim[0], img_dim[1],1)

        small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
                
        cut = small[20:40,:]
        
        info_section = np.zeros((10,cut.shape[1]),dtype=np.uint8) + 255
        info_section[9,:] = 0
        
        line = np.int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)
        
        if line != (0 or 100):
            info_section[:,line-1:line+2]  = 0
        elif line == 0:
            info_section[:,0:3]  = 0
        elif line == 100:
            info_section[:,info_section.shape[1]-3:info_section.shape[1]]  = 0
            
        total = np.concatenate((info_section, cut), axis=0)

        final_image = total.reshape( total.shape[0], total.shape[1],1)

        
            
        #cv2.imshow("Test", Image)
        #cv2.waitKey(0)
       

        return final_image


    def goalDirection(self, goal, pos):
        
        yaw  = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)[2]
        yaw = math.degrees(yaw) 
       
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
    
        return ((math.degrees(track) - 180) % 360) - 180 

    def takeAction(self, action):
        position = self.client.simGetVehiclePose().position
        yaw  = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)[2]
        if action == 0:
            yaw_change = 0.0
        elif action == 1:
            yaw_change = 30.0
        elif action == 2:
            yaw_change = -30.0

        action_yaw = yaw + yaw_change

        vx = math.cos(action_yaw)
        vy = math.sin(action_yaw)
        v_norm = np.linalg.norm([vx, vy])

        vx = vx / v_norm * self.MOVE_RATE
        vy = vy / v_norm * self.MOVE_RATE
        new_x = position.x_val + vx
        new_y = position.y_val + vy
        new_pose = airsim.Pose(airsim.Vector3r(new_x, new_y, self.HEIGHT), airsim.utils.to_quaternion(0, 0, action_yaw))
        self.client.simSetVehiclePose(new_pose, False)
       # time.sleep(self.CV_SLEEP_TIME)

        collided = self.checkCollision()
        
        return collided

    

    def checkCollision(self):
        collision_info = self.client.simGetCollisionInfo()
        return collision_info.has_collided

    def computeReward(self,pos):
        

        distance_now = np.sqrt(np.power((self.goal[0]-pos.x_val),2) + np.power((self.goal[1]-pos.y_val),2))

        distance_before = self.allLogs['distance'][-1]

        reward = -1

        reward = reward + (distance_before - distance_now)

        return reward,distance_now

    def step(self, action):
        
        collided = self.takeAction(action)

        current_position = self.client.simGetVehiclePose().position
        
        if collided == True:
            done = True
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]-current_position.x_val),2) + np.power((self.goal[1]-current_position.y_val),2))

         
        else: 
            done = False
            reward, distance = self.computeReward(current_position)

          
        
        # You made it
        if distance < 3:
            done = True
            reward = 100.0


        self.addToLog('distance', distance)

        self.addToLog('reward', reward)

        rewardSum = np.sum(self.allLogs['reward'])

        if rewardSum < -1000:
            done = True

        
        track = self.goalDirection(self.goal, current_position) 

        state = self.getDepth(track)


                

        return state, reward, done
