import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp

import airsim


class Environment():

    def __init__(self):        
        #self.img1 = None
        self.img2 = None

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
    
        #self.home_pos = self.client.simGetVehiclePose().position
    
        #self.home_ori = self.client.simGetVehiclePose().orientation

        self.z = -6
    
    def straight(self, duration, speed):
        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, airsim.DrivetrainType.ForwardOnly).join()

        start = time.time()
        return start, duration
    
    def yaw_right(self, duration):
        self.client.rotateByYawRateAsync(30, duration).join()

        start = time.time()
        return start, duration
    
    def yaw_left(self, duration):
        self.client.rotateByYawRateAsync(-30, duration).join()
        start = time.time()
        return start, duration
    
    
    def take_action(self, action):
		
        #check if copter is on level cause sometimes it goes up without a reason
        #x = 0
        #while self.client.simGetVehiclePose().position.z_val < -7.0:
        #    self.client.moveToZAsync(-6, 3).join()
        #   # time.sleep(1)
        #    print(self.client.simGetVehiclePose().position.z_val, "and", x)
        #    x = x + 1
        #    if x > 10:
        #        return True        
        
    
        start = time.time()
        duration = 0 
        
        collided = False

        if action == 0:
            # Move in direction of current heading with 4m/s for 1s
            start, duration = self.straight(1, 1)
        
            while duration > time.time() - start:
                if self.client.simGetCollisionInfo().has_collided == True:
                    return True    
                
            self.client.moveByVelocityAsync(0, 0, 0, 1).join()
            self.client.rotateByYawRateAsync(0, 1).join()
            
            
        if action == 1:
            #Rotate right with 30/s for 1s
            start, duration = self.yaw_right(0.8)
            
            while duration > time.time() - start:
                if self.client.simGetCollisionInfo().has_collided == True:
                    return True
            
            self.client.moveByVelocityAsync(0, 0, 0, 1).join()
            self.client.rotateByYawRateAsync(0, 1).join()
            
        if action == 2:
            #Rotate left with 30/s for 1s
            start, duration = self.yaw_left(1)
            
            while duration > time.time() - start:
                if self.client.simGetCollisionInfo().has_collided == True:
                    return True
                
            self.client.moveByVelocityAsync(0, 0, 0, 1).join()
            self.client.rotateByYawRateAsync(0, 1).join()
            
        return collided
    
    def goal_direction(self, goal, pos):
        
        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180    
    
    
    def getScreenDepthVis(self, track):

        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
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

       # cv2.imshow("Test", newImage1)
       # cv2.waitKey(0)
        '''
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
            
        cv2.imshow("Test", total)
        cv2.waitKey(0)
        '''
        return newImage1


    def AirSim_reset(self):
        
        self.client.reset()
        #time.sleep(0.2)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        #time.sleep(1)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(self.z, 3).join() 
        #time.sleep(3)


    def step(self,action):

        collided = self.take_action(action)

        new_state = self.getScreenDepthVis(5)

        reward = 5


        return new_state,reward,collided






        
   
