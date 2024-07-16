# Arm Dynamics Engine

from arm_dynamics_base import ArmDynamicsBase
import numpy as np
from geometry import rot, xaxis, yaxis


class ArmDynamicsStudent(ArmDynamicsBase):

    def dynamics_step(self, state, action, dt):
        # state has the following format: [q_0, ..., q_(n-1), qdot_0, ..., qdot_(n-1)] where n is the number of links
        # action has the following format: [mu_0, ..., mu_(n-1)]
        # You can make use of the additional variables:
        # self.num_links: the number of links
        # self.joint_viscous_friction: the coefficient of viscous friction
        # self.link_lengths: an array containing the lengths of all the links
        # self.link_masses: an array containing the masses of all the links
       
        # Unpack state variables
        q = self.get_q(state)                   # Joint positions
        q_dot = self.get_qd(state)              # Joint velocities
        theta = self.compute_theta(q)
        omega = self.compute_omega(q_dot)

        # Unpack action variables
        tau = action                            # Joint torques

        # Dynamics calculations
        g = 9.81                                # Gravitational acceleration

        # Initialize variables for recursive Newton-Euler algorithm
        m = self.link_masses
        l = self.link_lengths
        acc = np.zeros((self.num_links, 1))                         # Angular acceleration

        # Iteration
        def compute_q_dot(q_dot, acc, dt):
            for i in range(len(q_dot)):
                q_dot[i] += acc[i] * dt
            return q_dot

        def compute_q(q, q_dot, acc, dt):
            for i in range(len(q)):
                q[i] += q_dot[i] * dt + 0.5 * acc[i] * dt**2
            return q


        # For 1-link robot
        if self.num_links == 1:
                                    
            # Compute rotation matrix from i to w
            Rw = rot(theta[0])

            # Matrix A
            A = np.zeros((3,3))
            A[0,0] = 1
            A[1,1] = 1
            A[1,2] = -0.5*m[0]*l[0]
            A[2,1] = -0.5*l[0]
            A[2,2] = -(1/12)*m[0]*l[0]*l[0]

            # Matrix B
            B = np.zeros((3,1))
            B[0,0] = -0.5*m[0]*l[0]*q_dot[0]**2 + m[0]*g*Rw[0,1]
            B[1,0] = m[0]*g*Rw[1,1]
            B[2,0] = self.joint_viscous_friction*q_dot[0] - tau[0]

            # Solve for x
            x = np.linalg.solve(A, B) 
            
            # Iteration
            acc = x[2]
            q_dot = compute_q_dot(q_dot, acc, dt)
            q = compute_q(q, q_dot, acc, dt)


        # For 2-link robot
        if self.num_links == 2:

            # Initialize Matrix A and B
            A = np.zeros((8,8))
            B = np.zeros((8,1))

            # For the first link

            # Compute rotation matrix from i to i+1
            R_i_ip1 = np.transpose(rot(q[1]))

            # Compute rotation matrix from i to w
            Rw = rot(theta[0])
                
            # Matrix A
            A[0,:8] = [1, 0, 0, -R_i_ip1[0,0], -R_i_ip1[0,1], 0, 0, 0]
            A[1,:8] = [0, 1, -0.5*m[0]*l[0], -R_i_ip1[1,0], -R_i_ip1[1,1], 0, 0, 0]
            A[2,:8] = [0, -0.5*l[0], -(1/12)*m[0]*l[0]**2, -0.5*l[0]*R_i_ip1[1,0], -0.5*l[0]*R_i_ip1[1,1], 0, 0, 0]

            # Matrix B
            B[0,0] = -0.5*m[0]*l[0]*omega[0]**2 + m[0]*g*Rw[0,1]
            B[1,0] = m[0]*g*Rw[1,1]
            B[2,0] = self.joint_viscous_friction*q_dot[0] + tau[1] - tau[0]

            # For the last link

            # Compute rotation matrix from i to i-1
            R_i_im1 = rot(q[-1])

            # Compute rotation matrix from i to w
            Rw = rot(theta[-1])   
                     
            # Matrix A
            A[-5,-8:] = [0, 0, 0, 1, 0, -m[-1], 0, 0]
            A[-4,-8:] = [0, 0, 0, 0, 1, 0, -m[-1], -0.5*l[-1]*m[-1]]
            A[-3,-8:] = [0, 0, 0, 0, -0.5*l[1], 0, 0, -(1/12)*m[-1]*l[-1]**2]
            A[-2,-8:] = [0, 0, l[-2]*R_i_im1[0,1], 0, 0, -1, 0, 0]
            A[-1,-8:] = [0, 0, l[-2]*R_i_im1[1,1], 0, 0, 0, -1, 0]

            # Matrix B
            B[-5,0] = -0.5*m[-1]*l[-1]*omega[-1]**2 + m[-1]*g*Rw[0,1]
            B[-4,0] = m[-1]*g*Rw[1,1]
            B[-3,0] = self.joint_viscous_friction*q_dot[-1] - tau[-1]
            B[-2,0] = l[-2]*R_i_im1[0,0]*omega[-2]**2
            B[-1,0] = l[-2]*R_i_im1[1,0]*omega[-2]**2

            # Solve for x
            x = np.linalg.solve(A, B)  

            # Iteration
            acc[0] = x[2]
            acc[-1] = x[-1] - x[-6] 
            q_dot = compute_q_dot(q_dot, acc, dt)
            q = compute_q(q, q_dot, acc, dt)


        # For 3-link robot
        if self.num_links == 3:
            A = np.zeros((13,13))       # Matrix A (A*x=B)
            B = np.zeros((13,1))        # Matrix B (A*x=B)
               
            # For the first link
            # Compute rotation matrix from i to i+1
            R_i_ip1 = np.transpose(rot(q[1]))

            # Compute rotation matrix from i to w
            Rw = rot(theta[0])
                
            # Matrix A
            A[0,:] = [1, 0, -R_i_ip1[0,0], -R_i_ip1[0,1], 0, 0, 0, 0, 0, 0, 0, 0, 0]
            A[1,:] = [0, 1, -R_i_ip1[1,0], -R_i_ip1[1,1], 0, 0, -0.5*m[0]*l[0], 0, 0, 0, 0, 0, 0]
            A[6,:] = [0, -0.5*l[0], -0.5*l[0]*R_i_ip1[1,0], -0.5*l[0]*R_i_ip1[1,1], 0, 0, -(1/12)*m[0]*l[0]**2, 0, 0, 0, 0, 0, 0]

            # Matrix B
            B[0,0] = -0.5*m[0]*l[0]*omega[0]**2 + m[0]*g*Rw[0,1]
            B[1,0] = m[0]*g*Rw[1,1]
            B[6,0] = self.joint_viscous_friction*q_dot[0] + tau[1] - tau[0]

            # For the second link
            # Compute rotation matrix from i to i-1
            R_i_im1 = rot(q[1])

            # Compute rotation matrix from i to i+1
            R_i_ip1 = np.transpose(rot(q[2]))

            # Compute rotation matrix from i to w
            Rw = rot(theta[1])

            # Matrix A
            A[2,:] = [0, 0, 1, 0, -R_i_ip1[0,0], -R_i_ip1[0,1], 0, 0, 0, -m[1], 0, 0, 0]
            A[3,:] = [0, 0, 0, 1, -R_i_ip1[1,0], -R_i_ip1[1,1], 0, -0.5*m[1]*l[1], 0, 0, - m[1], 0, 0]
            A[7,:] = [0, 0, 0, -0.5*l[1], -0.5*l[1]*R_i_ip1[1,0], -0.5*l[1]*R_i_ip1[1,1], 0, -(1/12)*m[1]*l[1]**2, 0, 0, 0, 0, 0]
            A[9,:] = [0, 0, 0, 0, 0, 0, l[0]*R_i_im1[0,1], 0, 0, -1, 0, 0, 0]
            A[10,:] = [0, 0, 0, 0, 0, 0, l[0]*R_i_im1[1,1], 0, 0, 0, -1, 0, 0]
            
            # Matrix B
            B[2,0] = -0.5*m[1]*l[1]*omega[1]**2 + m[1]*g*Rw[0,1]
            B[3,0] = m[1]*g*Rw[1,1]
            B[7,0] = self.joint_viscous_friction*q_dot[1] + tau[2] - tau[1]
            B[9,0] = l[0]*R_i_im1[0,0]*omega[0]**2
            B[10,0] = l[0]*R_i_im1[1,0]*omega[0]**2

            # For the last link
            # Compute rotation matrix from i to i-1
            R_i_im1 = rot(q[-1])

            # Compute rotation matrix from i to w
            Rw = rot(theta[-1])   
                    
            # Matrix A
            A[4,:] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -m[-1], 0]
            A[5,:] = [0, 0, 0, 0, 0, 1, 0, 0, -0.5*l[-1]*m[-1], 0, 0, 0, -m[-1]]
            A[8,:] = [0, 0, 0, 0, 0, -0.5*l[-1], 0, 0, -(1/12)*m[-1]*l[-1]**2, 0, 0, 0, 0]
            A[11,:] = [0, 0, 0, 0, 0, 0, 0, l[-2]*R_i_im1[0,1], 0, R_i_im1[0,0], R_i_im1[0,1], -1, 0]
            A[12,:] = [0, 0, 0, 0, 0, 0, 0, l[-2]*R_i_im1[1,1], 0, R_i_im1[1,0], R_i_im1[1,1], 0, -1]

            # Matrix B
            B[4,0] = -0.5*m[-1]*l[-1]*omega[-1]**2 + m[-1]*g*Rw[0,1]
            B[5,0] = m[-1]*g*Rw[1,1]
            B[8,0] = self.joint_viscous_friction*q_dot[-1] - tau[-1]
            B[11,0] = l[-2]*R_i_im1[0,0]*omega[-2]**2
            B[12,0] = l[-2]*R_i_im1[1,0]*omega[-2]**2

            # Solve for x
            x = np.linalg.solve(A, B)     

            # Iteration
            acc[0] = x[6]
            acc[1] = x[7] - x[6] 
            acc[2] = x[8] - x[7]
            q_dot[0] += acc[0] * dt
            q_dot[1] += acc[1] * dt
            q_dot[2] += acc[2] * dt
            q[0] += q_dot[0] * dt + 0.5 * acc[0] * dt**2
            q[1] += q_dot[1] * dt + 0.5 * acc[1] * dt**2
            q[2] += q_dot[2] * dt + 0.5 * acc[2] * dt**2


        # # For n-links
        # if self.num_links > 3:
        #     A = np.zeros((self.num_links*5-2,self.num_links*5-2))       # Matrix A (A*x=B)
        #     B = np.zeros((self.num_links*5-2,1))                        # Matrix B (A*x=B)

        #     for i in range(self.num_links):
                
        #         # For the first link
        #         if i == 0:
        #             # Compute rotation matrix from i to i+1
        #             R_i_ip1 = np.transpose(rot(q[1]))

        #             # Compute rotation matrix from i to w
        #             Rw = rot(theta[0])
                        
        #             # Matrix A
        #             A[0,:8] = [1, 0, 0, -R_i_ip1[0,0], -R_i_ip1[0,1], 0, 0, 0]
        #             A[1,:8] = [0, 1, -0.5*m[0]*l[0], -R_i_ip1[1,0], -R_i_ip1[1,1], 0, 0, 0]
        #             A[2,:8] = [0, -0.5*l[0], -(1/12)*m[0]*l[0]**2, -0.5*l[0]*R_i_ip1[1,0], -0.5*l[0]*R_i_ip1[1,1], 0, 0, 0]

        #             # Matrix B
        #             B[0,0] = -0.5*m[0]*l[0]*omega[0]**2 + m[0]*g*Rw[0,1]
        #             B[1,0] = m[0]*g*Rw[1,1]
        #             B[2,0] = self.joint_viscous_friction*q_dot[0] + tau[1] - tau[0]

        #         # For the second link
        #         if i == 1:
        #             # Compute rotation matrix from i to i-1
        #             R_i_im1 = rot(q[i])

        #             # Compute rotation matrix from i to i+1
        #             R_i_ip1 = np.transpose(rot(q[i+1]))

        #             # Compute rotation matrix from i to w
        #             Rw = rot(theta[i])

        #             # Matrix A
        #             A[3,:13] = [0, 0, 0, 1, 0, -m[i], 0, 0, -R_i_ip1[0,0], -R_i_ip1[0,1], 0, 0, 0]
        #             A[4,:13] = [0, 0, 0, 0, 1, 0, -m[i], -0.5*m[i]*l[i], -R_i_ip1[1,0], -R_i_ip1[1,1], 0, 0, 0]
        #             A[5,:13] = [0, 0, 0, 0, -0.5*l[i], 0, -m[i], -(1/12)*m[i]*l[i]**2, -0.5*l[i]*R_i_ip1[1,0], -0.5*l[i]*R_i_ip1[1,1], 0, 0, 0]
        #             A[6,:13] = [0, 0, l[i-1]*R_i_im1[0,1], 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]
        #             A[7,:13] = [0, 0, l[i-1]*R_i_im1[1,1], 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]
                    
        #             # Matrix B
        #             B[3,0] = -0.5*m[i]*l[i]*omega[i]**2 + m[i]*g*Rw[0,1]
        #             B[4,0] = m[i]*g*Rw[1,1]
        #             B[5,0] = self.joint_viscous_friction*q_dot[i] + tau[i+1] - tau[i]
        #             B[6,0] = l[i-1]*R_i_im1[0,0]*omega[i-1]**2
        #             B[7,0] = l[i-1]*R_i_im1[1,0]*omega[i-1]**2

        #         # For the intermediate links
        #         if self.num_links > 3:
        #             # Mapping
        #             index_r = (i-1)*5 + 3
        #             index_c = (i-1)*5 - 2

        #             # Compute rotation matrix from i to i-1
        #             R_i_im1 = rot(q[i])

        #             # Compute rotation matrix from i to i+1
        #             R_i_ip1 = np.transpose(rot(q[i+1]))

        #             # Compute rotation matrix from i to w
        #             Rw = rot(theta[i])

        #             # Matrix A
        #             A[index_r+0,index_c:index_c+15] = [0, 0, 0, 0, 0, 1, 0, -m[i], 0, 0, -R_i_ip1[0,0], -R_i_ip1[0,1], 0, 0, 0]
        #             A[index_r+1,index_c:index_c+15] = [0, 0, 0, 0, 0, 0, 1, 0, -m[i], -0.5*m[i]*l[i], -R_i_ip1[1,0], -R_i_ip1[1,1], 0, 0, 0]
        #             A[index_r+2,index_c:index_c+15] = [0, 0, 0, 0, 0, 0, -0.5*l[i], 0, -m[i], -(1/12)*m[i]*l[i]**2, -0.5*l[i]*R_i_ip1[1,0], -0.5*l[i]*R_i_ip1[1,1], 0, 0, 0]
        #             A[index_r+3,index_c:index_c+15] = [0, 0, R_i_im1[0,0], R_i_im1[0,1], l[i-1]*R_i_im1[0,1], 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]
        #             A[index_r+4,index_c:index_c+15] = [0, 0, R_i_im1[1,0], R_i_im1[1,1], l[i-1]*R_i_im1[1,1], 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]
                    
        #             # Matrix B
        #             B[index_r+0,0] = -0.5*m[i]*l[i]*omega[i]**2 + m[i]*g*Rw[0,1]
        #             B[index_r+1,0] = m[i]*g*Rw[1,1]
        #             B[index_r+2,0] = self.joint_viscous_friction*q_dot[i] + tau[i+1] - tau[i]
        #             B[index_r+3,0] = l[i-1]*R_i_im1[0,0]*omega[i-1]**2
        #             B[index_r+4,0] = l[i-1]*R_i_im1[1,0]*omega[i-1]**2

        #         # For the last link
        #         if i == self.num_links:
        #             # Compute rotation matrix from i to i-1
        #             R_i_im1 = rot(q[-1])

        #             # Compute rotation matrix from i to w
        #             Rw = rot(theta[-1])   
                            
        #             # Matrix A
        #             A[-5,-8:] = [0, 0, 0, 1, 0, -m[-1], 0, 0]
        #             A[-4,-8:] = [0, 0, 0, 0, 1, 0, -m[-1], -0.5*l[-1]*m[-1]]
        #             A[-3,-8:] = [0, 0, 0, 0, -0.5*l[1], 0, 0, -(1/12)*m[-1]*l[-1]**2]
        #             A[-2,-8:] = [0, 0, l[-2]*R_i_im1[0,1], 0, 0, -1, 0, 0]
        #             A[-1,-8:] = [0, 0, l[-2]*R_i_im1[1,1], 0, 0, 0, -1, 0]

        #             # Matrix B
        #             B[-5,0] = -0.5*m[-1]*l[-1]*omega[-1]**2 + m[-1]*g*Rw[0,1]
        #             B[-4,0] = m[-1]*g*Rw[1,1]
        #             B[-3,0] = self.joint_viscous_friction*q_dot[-1] - tau[-1]
        #             B[-2,0] = l[-2]*R_i_im1[0,0]*omega[-2]**2
        #             B[-1,0] = l[-2]*R_i_im1[1,0]*omega[-2]**2

        #         # Solve for x
        #         x = np.linalg.solve(A, B)     

        #         # Iteration
        #         acc[0] = x[2]
        #         acc[-1] = x[-1] - x[-6]
        #         if self.num_links == 3:
        #             acc[1] = x[7] - acc[0] 
        #         elif self.num_links > 3:
        #             for i in range(self.num_links-1):
        #                 if i > 0:
        #                     index_r = (i-1)*5 + 2
        #                     acc[i-1] = x[index_r+5] - acc[i]
        #         q_dot = compute_q_dot(q_dot, acc, dt)
        #         q = compute_q(q, q_dot, acc, dt)


        # Update state
        state = np.concatenate((q, q_dot))

        return state

    
