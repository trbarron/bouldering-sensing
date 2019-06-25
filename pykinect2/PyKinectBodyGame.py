from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import pygame
import sys
import math
import numpy as np

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.Color("#FCE4EC"),  
                  pygame.Color("#F8BBD0"),
                  pygame.Color("#F48FB1"),
                  pygame.Color("#F06292"),
                  pygame.Color("#EC407A"),
                  pygame.Color("#E91E63"),
                  pygame.Color("#D81B60"),
                  pygame.Color("#C2185B"),
                  pygame.Color("#AD1457"),
                  pygame.Color("#880E4F"),
                  pygame.Color("#880E4F")]

class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("hold highlighting demo")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        self._old_bodies = []

        self._flashback_timer = 0

    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)
        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);

        

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()


    def find_nearby_pixels(self,array,mouse_color,scalar,intended_size_x,intended_size_y,difference,x,y):

        movement = [[0,1],[0,-1],[1,0],[-1,0]]
        for move in movement:
            new_location = (x+move[0]*scalar,y+move[1]*scalar)
            if new_location not in array and new_location[0] > 0 and new_location[0] < intended_size_x and new_location[1] > 0 and new_location[1] < intended_size_y:
                _pos = (new_location[0],new_location[1])

                pixel_color = self._frame_surface.get_at(_pos)

                sum_err = 0
                r_bar = (pixel_color[0] + mouse_color[0]) / 2
                r_delt = (pixel_color[0] - mouse_color[0])
                g_delt = (pixel_color[1] - mouse_color[1])
                b_delt = (pixel_color[2] - mouse_color[2])


                delt = (2+r_bar/256)*r_delt**2 + 4*g_delt**2 + (2+(255-r_bar)/256)*b_delt**2
                if delt < difference:
                         array.add(_pos)
                         self.find_nearby_pixels(array,mouse_color,scalar,intended_size_x,intended_size_y,difference,new_location[0],new_location[1])
        return array

    def highlight_color(self,mouse_color,mouse_pos):

        intended_size_x, intended_size_y = self._frame_surface.get_size()
        actual_size_x,actual_size_y = pygame.display.get_surface().get_size()

        num = 0

        array = set()
        scalar = 5
        difference = 1100

        initial_x,initial_y = mouse_pos

        initial_x = math.floor(initial_x/scalar)*scalar
        initial_y = math.floor(initial_y/scalar)*scalar

        array = self.find_nearby_pixels(array,mouse_color,scalar,intended_size_x,intended_size_y,difference,initial_x,initial_y)

        #Get depth of your array of points
        _depth_frame = self._kinect.get_last_depth_frame()
        _depth_width = self._kinect.depth_frame_desc.width
        _depth_height = self._kinect.depth_frame_desc.height



        _organized_depth_frame = []
        for w in range(_depth_width):
            _organized_depth_frame.append(_depth_frame[w*_depth_height:(w+1)*_depth_height])

        depth_array = []
        for point in array:
            x,y = point
            relx,rely = x/intended_size_x,y/intended_size_y
            depthx,depthy = math.floor(relx*_depth_width),math.floor(rely*_depth_height)
            depth = 0
            depth = _organized_depth_frame[depthx][depthy]
            if depth > 4500: depth = 0
            #depth_array.append((x,y,depth))
            depth_array.append(depth)
        
        avg_depth = np.median(depth_array)    
        avg_array = []

        for point in array:
            x,y = point
            avg_array.append((x,y,avg_depth))

        #print(min(depth_array))
        #print(max(depth_array))
        #print(np.mean(depth_array))
        #print(np.median(depth_array))

        return avg_array

    def run(self):
        file_num = 0
        mouse_color = None
        hold_array = []


        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)   
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_SPACE]:
                        mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
                        intended_size_x, intended_size_y = self._frame_surface.get_size()
                        
                        actual_size_x,actual_size_y = pygame.display.get_surface().get_size()


                        rel_mouse_x = mouse_pos_x / actual_size_x
                        rel_mouse_y = mouse_pos_y / actual_size_y

                        mouse_pos = (int(rel_mouse_x*intended_size_x),
                                     int(rel_mouse_y*intended_size_y))

                        mouse_color = self._frame_surface.get_at(mouse_pos)
                        new_hold = self.highlight_color(mouse_color,mouse_pos)

                        commonality = False
                        for hold in hold_array:
                            hold = hold[0]
                            #see if common element
                            common_elements = list(set(hold).intersection(new_hold))
                            if len(common_elements) > 0:
                                for new_element in new_hold:
                                    hold.append(new_element)
                                commonality = True
                        if not commonality: hold_array.append([new_hold,False])
                elif event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_r]:
                        hold_array = []
                    if keys[pygame.K_p]:
                        if len(hold_array) > 0:
                            hold_array.pop()


                    
            # --- Game logic should go here




            # --- Getting frames and drawing  
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            #refresh_rate = 5
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            if mouse_color is not None:
                pygame.draw.rect(self._frame_surface, mouse_color, [0, 0, 100, 100], 0)
                
            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()


            scalar = 5
            # --- draw skeletons to _frame_surface
            if self._bodies is not None: 
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: continue

                    joints = body.joints 

                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)

                    self.draw_body(joints, joint_points, SKELETON_COLORS[0])

                    rh = joints[PyKinectV2.JointType_HandRight]
                    rh_pos = self._kinect.body_joint_to_color_space(rh)
                    rh_depth = joints[PyKinectV2.JointType_HandRight].Position.z
                    if rh_depth > 4.5: rh_depth = 4.5
                    if float('inf') not in [rh_pos.x,rh_pos.y] and-1*float('inf') not in [rh_pos.x,rh_pos.y]: 
                        rh_pos = ((rh_pos.x),(rh_pos.y),(rh_depth)) 
                    else: rh_pos = (0,0,0)
                    #lh_pos = ((joints[PyKinectV2.JointType_HandLeft].Position.x),(joints[PyKinectV2.JointType_HandLeft].Position.y),(joints[PyKinectV2.JointType_HandLeft].Position.z))
                    rh_bucket = (math.floor(rh_pos[0]/scalar)*scalar,math.floor(rh_pos[1]/scalar)*scalar)

                    for hold_i in range(len(hold_array)):
                        hold = hold_array[hold_i][0]
                        for pos in hold:
                            dist = ((rh_bucket[0]-pos[0])**2 + (rh_bucket[1]-pos[1])**2)
                            if dist < 4000:
                                #print("dist: ", dist)
                                #print("hand: ",rh_pos[2])
                                #print("init: ",float(pos[2]/1000))
                                #print("delt: ", abs(rh_pos[2] - pos[2]/1000))
                                if abs(rh_pos[2] - pos[2]/1000) < 0.6:

                                    hold_array[hold_i][1] = True

                    lh = joints[PyKinectV2.JointType_HandLeft]
                    lh_pos = self._kinect.body_joint_to_color_space(lh)
                    lh_depth = joints[PyKinectV2.JointType_HandLeft].Position.z
                    if lh_depth > 4.5: lh_depth = 4.5
                    if float('inf') not in [lh_pos.x,lh_pos.y] and-1*float('inf') not in [lh_pos.x,lh_pos.y]: 
                        lh_pos = ((lh_pos.x),(lh_pos.y),(lh_depth)) 
                    else: lh_pos = (0,0,0)
                    #lh_pos = ((joints[PyKinectV2.JointType_HandLeft].Position.x),(joints[PyKinectV2.JointType_HandLeft].Position.y),(joints[PyKinectV2.JointType_HandLeft].Position.z))
                    lh_bucket = (math.floor(lh_pos[0]/scalar)*scalar,math.floor(lh_pos[1]/scalar)*scalar)

                    for hold_i in range(len(hold_array)):
                        hold = hold_array[hold_i][0]
                        for pos in hold:
                            dist = ((lh_bucket[0]-pos[0])**2 + (lh_bucket[1]-pos[1])**2)
                            if dist < 4000:
                                #print("dist: ", dist)
                                #print("hand: ",rh_pos[2])
                                #print("init: ",float(pos[2]/1000))
                                #print("delt: ", abs(rh_pos[2] - pos[2]/1000))
                                if abs(lh_pos[2] - pos[2]/1000) < 0.6:

                                    hold_array[hold_i][1] = True


                    self._old_bodies.append(body)
                    self._old_bodies = self._old_bodies[-10:]

                    color_num = 0
                    for old_body in self._old_bodies:
                        color_num += 1
                        joints = old_body.joints 
                        joint_points = self._kinect.body_joints_to_color_space(joints)
                        self.draw_body(joints, joint_points, SKELETON_COLORS[color_num])

                    self._flashback_timer = (self._flashback_timer+1)%10
    

            #draw color blobs
            for hold in hold_array:
                touched_flag = hold[1]
                hold = hold[0]
                for pos in hold:
                    clr = 255 if touched_flag else 100
                    pygame.draw.circle(self._frame_surface, pygame.Color(clr, clr, clr, 230),pos[:2],3)

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))

            surface_to_draw = None
           


            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # Save every frame

            # --- Limit to 60 frames per second
            self._clock.tick(10)

            pygame.display.update()

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "aww shit"
game = BodyGameRuntime();
game.run();

