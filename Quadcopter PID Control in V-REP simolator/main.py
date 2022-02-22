#!/usr/bin/python
import math
import numpy as np
import matplotlib.pyplot as plt
import vrep
import rl_helper

def deg2rad(angle):return angle*3.1415/180.0
def rad2deg(angle):return angle*180.0/3.1415
def cos(angle):return math.cos(angle)
def sin(angle):return math.sin(angle)
def bodyFrameAngle(p,q,r):
    transformMatrix_I2B = np.array([[cos(q)*cos(r),cos(q)*sin(r),-sin(q)],
                                   [sin(p)*sin(q)*cos(r)-cos(p)*sin(r),sin(p)*sin(q)*sin(r)+cos(p)*cos(r),sin(p)*cos(q)],
                                   [cos(p)*sin(q)*cos(r)+sin(p)*sin(r),cos(p)*sin(q)*sin(r)-sin(p)*cos(r),cos(p)*cos(q)]])
    pqr = np.array([[p],[q],[r]])
    out = transformMatrix_I2B@pqr
    return [out[0][0],out[1][0],out[2][0]]

mg   = 5.0
dt   = 0.01

Kp_x, Ki_x, Kd_x  = 1.0, 0.4, 0.4
Kp_y, Ki_y, Kd_y  = 1.0, 0.4, 0.4
Kp_z, Ki_z, Kd_z  = 4.0, 1, 2
Kp_phi, Ki_phi, Kd_phi = -0.7, -0.5, -0.5
Kp_tta, Ki_tta, Kd_tta = 0.7, 0.5, 2.0
Kp_psi, Ki_psi, Kd_psi = 3, 1, 2

ei_x, ei_y, ei_z = 0, 0, 0
ei_phi, ei_tta, ei_psi = 0, 0, 0

lastEp_x, lastEp_y, lastEp_z = 1, 1, 1
lastEp_phi, lastEp_tta, lastEp_psi = 0, 0, 0

b, l, d = 0.001, 0.1, 0.0001

Xs = np.array([[0,0,0,0,0,0]])

rl_functions = None
try:
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

    if clientID != -1:
        rl_functions = rl_helper.RL(clientID)
        print('Main Script Started')

        rl_functions.init_sensors()
        rl_functions.synchronous(enable=True)
        rl_functions.start_sim()

        while vrep.simxGetConnectionId(clientID) != -1:
            
            # target position and attitude
            targetPos = [1,2,3]
            targetAttitude = [0,0,0]
            
            # set target position
            rl_functions.setTargetPosition(targetName='Quadricopter_target', desiredPos=targetPos)
            
            rl_functions.target_z = targetPos[2]
            pos = rl_functions.get_position()
            x, y, z = pos[0], pos[1], pos[2]
            
            ep_x = targetPos[0] - x; ei_x = ei_x+ep_x*dt; ed_x = (ep_x-lastEp_x)/dt
            ep_y = targetPos[1] - y; ei_y = ei_y+ep_y*dt; ed_y = (ep_y-lastEp_y)/dt
            ep_z = targetPos[2] - z; ei_z = ei_z+ep_z*dt; ed_z = (ep_z-lastEp_z)/dt
            
            # Vertical control:
            U1 = 4*b*(mg + Kp_z*ep_z + Ki_z*ei_z + Kd_z*ed_z)
            
            # Save errors
            lastEp_x = ep_x 
            lastEp_y = ep_y 
            lastEp_z = ep_z   
            
            # Rotational control:
            attitude=rl_functions.get_attitude()
            phi, tta, psi = attitude[0], attitude[1], attitude[2] 
            # [phi,tta,psi] = bodyFrameAngle(phi,tta,psi)
            
            # Desired phi and tta
            phi_d = 1.1*math.atan(Kp_y*ep_y+Ki_y*ei_y+Kd_y*ed_y) #deg2rad(Kp_y*ep_y+Ki_y*ei_y+Kd_y*ed_y)
            tta_d = math.atan(Kp_x*ep_x+Ki_x*ei_x+Kd_x*ed_x) #deg2rad(Kp_x*ep_x+Ki_x*ei_x+Kd_x*ed_x)
            
            # Attitude control: 
            ep_phi = -phi_d - phi ; ei_phi = ei_phi+ep_phi*dt; ed_phi = (ep_phi-lastEp_phi)/dt
            ep_tta = tta_d - tta ; ei_tta = ei_tta+ep_tta*dt; ed_tta = (ep_tta-lastEp_tta)/dt
            ep_psi = targetAttitude[2] - psi ; ei_psi = ei_psi+ep_psi*dt; ed_psi = (ep_psi-lastEp_psi)/dt
            
            U2 = 2*b*l*(Kp_phi*ep_phi + Ki_phi*ei_phi + Kd_phi*ed_phi)
            U3 = 2*b*l*(Kp_tta*ep_tta + Ki_tta*ei_tta + Kd_tta*ed_tta)
            U4 = 4*d*(Kp_psi*ep_psi + Ki_psi*ei_psi + Kd_psi*ed_psi)

            lastEp_phi = ep_phi
            lastEp_tta = ep_tta
            lastEp_psi = ep_psi
            
            Omega1_pow2 = 1/(4*b)*U1-1/(2*b*l)*U3-1/(4*d)*U4
            Omega2_pow2 = 1/(4*b)*U1-1/(2*b*l)*U2+1/(4*d)*U4
            Omega3_pow2 = 1/(4*b)*U1+1/(2*b*l)*U3-1/(4*d)*U4
            Omega4_pow2 = 1/(4*b)*U1+1/(2*b*l)*U2+1/(4*d)*U4
            
            rl_functions.rotor_data = [Omega1_pow2, Omega2_pow2, Omega3_pow2, Omega4_pow2]
            rl_functions.do_action()
            
            if abs(x)>10.0 or abs(y)>10.0 or abs(z)>5.0:
                rl_functions.stop_sim()
                break
            
            # print(rl_functions.get_reward())
            print("x:"+"{:.4f}".format(x)+", y:"+"{:.4f}".format(y)+", z:"+"{:.4f}".format(z)+
                  ", phi:"+"{:.4f}".format(rad2deg(phi))+", tta:"+"{:.4f}".format(rad2deg(tta))+
                  ", psi:"+"{:.4f}".format(rad2deg(psi)))
            
            Xs = np.append(Xs,[[x,y,z,rad2deg(phi),rad2deg(tta),rad2deg(psi)]],axis=0)
            rl_functions.synchronousTrigger()
    else:
        print("Failed to connect to remote API Server")
        rl_functions.stop_sim()
        vrep.simxFinish(clientID)
except KeyboardInterrupt:
    rl_functions.stop_sim()
    vrep.simxFinish(clientID)
    
#%% Plot results

plt.figure()
plt.plot(Xs[:,0], label="x")
plt.plot(Xs[:,1], label="y")
plt.plot(Xs[:,2], label="z")
plt.legend(loc="upper right")
plt.grid()
plt.show()

plt.figure()
plt.plot(Xs[:,3], label=r"$\phi$")
plt.plot(Xs[:,4], label=r"$\theta$")
plt.plot(Xs[:,5], label=r"$\psi$")
plt.legend(loc="upper right")
plt.grid()
plt.show()
