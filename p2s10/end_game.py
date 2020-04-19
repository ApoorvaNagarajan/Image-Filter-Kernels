# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai import TD3

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3(5,1,5)
last_reward = 0
scores = []

crop_size = 80
border_size = 5

# Initializing the map
first_update = True
def init():
    global sand
    global img
    global goal_x
    global goal_y
    global first_update
    
    # Read the mask image
    sand = np.zeros((map_height,map_width))
    img = cv2.imread("./images/MASK1.png",0) 
    sand = img/255
          
    goal_x = 1197
    goal_y = 512
    first_update = False
    global swap
    swap = 0
    global done_flag
    done_flag = 0
    global total_timesteps
    total_timesteps = 0


# Initializing the last distance
last_distance = 0

max_episode_timesteps = 2000

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0.0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = float(rotation)
        self.angle = (self.angle + self.rotation)%360
        
    def reset(self):
        print("RESETTING")
        self.pos.x = np.random.randint(80, map_width-80, size=1)
        self.pos.y = np.random.randint(80, map_height-80, size=1)
        

class Goal(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        
        
    def get_surroundings(self):
        
        crop_img = sand[map_height-int(self.car.y)-crop_size: map_height- int(self.car.y)+crop_size, int(self.car.x)-crop_size:int(self.car.x)+crop_size].copy()
       
        top = 0
        bottom = 0
        left = 0
        right = 0
         
        # if at frame boundary, pad the cropped image with sand (1's)
        if(crop_img.shape[0] != 2*crop_size): # rows
            if(self.car.pos[1] < crop_size):
                bottom = 2*crop_size - crop_img.shape[0]
            else:
                top = 2*crop_size - crop_img.shape[0]
            
        if(crop_img.shape[1] != 2*crop_size): # colums
            if(self.car.pos[0] < crop_size):
                left = 2*crop_size - crop_img.shape[1]
            else:
                right = 2*crop_size - crop_img.shape[1]            
            
        if((top != 0) or (bottom != 0) or (left != 0) or (right != 0)):
            print(crop_size, top, bottom, right, left)
            crop_img = cv2.copyMakeBorder(crop_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1 )    
        #cv2.imshow("crop_img",crop_img)
        #cv2.waitKey(0)  

        pt1 = Vector(30, 0).rotate(-self.car.angle) + (crop_size, crop_size)
        pt2 = Vector(0, 10).rotate(-self.car.angle) + (crop_size, crop_size)
        pt3 = Vector(0, -10).rotate(-self.car.angle) + (crop_size, crop_size)
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        ctr = np.array(triangle_cnt).reshape((-1,3,2)).astype(np.int32)
        cv2.fillPoly(crop_img, pts =[ctr], color=0.5)     
        #cv2.imshow("Car",crop_img)
        #cv2.waitKey(0) 
        
        rsz_img = cv2.resize(crop_img, (32,32), interpolation = cv2.INTER_AREA)
        #cv2.imshow("resized_image",rsz_img)
        #cv2.waitKey(0) 
        
        rsz_img = rsz_img.reshape(1, 32, 32)

        return rsz_img

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global map_width
        global map_height
        global swap
        global done_flag
        global total_timesteps
        

        map_width = self.width
        map_height = self.height
        if first_update:
            init()
            self.surr = self.get_surroundings()
            print(self.car.x, self.car.y)


        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        
        
        # states : 
        #32x32 cropped image with car overlay
        #orientation
        #-orientation
        #distance_x from goal
        #distance_y from goal       
        X1 = self.surr       
        X2 = [orientation, -orientation, xx, yy]

        # actions:
        # angle theta of rotation       
        action = brain.select_action(X1, X2)
        print(action)
        self.car.move(action[0]) 
               
        if self.car.x < border_size:
            self.car.x = border_size
            last_reward = -10
            done_flag = 1
        if self.car.x > self.width - border_size:
            self.car.x = self.width - border_size
            last_reward = -10
            done_flag = 1
        if self.car.y < border_size:
            self.car.y = border_size
            last_reward = -10
            done_flag = 1
        if self.car.y > self.height - border_size:
            self.car.y = self.height - border_size
            last_reward = -10
            done_flag = 1
            
        on_road = 0
        
        if(0 == done_flag):

            # velocity
            if sand[map_height-int(self.car.y), int(self.car.x)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                on_road = 0
                print("SAND")
            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                on_road = 1
                print("ROAD")
            
            new_xx = goal_x - self.car.x
            new_yy = goal_y - self.car.y
            new_orient = Vector(*self.car.velocity).angle((new_xx,new_yy))/180.
            new_X1 = self.get_surroundings()
            new_X2 = [new_orient, -new_orient, new_xx, new_yy]
            self.surr = new_X1
                            
            # Rewards
            distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
            
            if((on_road == 1) and (distance < last_distance)):
                last_reward = 1
            elif((on_road == 0) and (distance < last_distance)):
                last_reward = -2
            elif((on_road == 1) and (distance > last_distance)):
                last_reward = -1
            elif((on_road == 0) and (distance > last_distance)):
                last_reward = -4
            
        else:
            
            # Rewards
            distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
            new_X1 = X1
            new_X2 = X2


        if distance < 25:
            print("GOALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL REACHEDDDDDDDDDDDDDDDDDD")
            if swap == 1:
                goal_x = 1197
                goal_y = 512
                swap = 0
                done_flag = 1
                #self.goal.pos = 1420, 622
            else:
                goal_x = 361
                goal_y = 311
                swap = 1
                done_flag = 1
                self.goal.pos = 9, 85
                
        last_distance = distance

        done_flag = brain.add_replay_buff(X1, X2, new_X1, new_X2, action, last_reward, done_flag)

        if(done_flag == 1):
            self.car.reset()
            self.surr = self.get_surroundings()
            done_flag = 0

        total_timesteps += 1

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.y), int(touch.x)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.y) - 10 : int(touch.y) + 10, int(touch.x) - 10 : int(touch.x) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((map_height,map_width))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
