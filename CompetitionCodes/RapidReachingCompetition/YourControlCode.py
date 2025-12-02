import mujoco
import numpy as np
from itertools import permutations


class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData, target_points):
    self.m = m
    self.d = d
    self.target_points = target_points

    self.init_qpos = d.qpos.copy()

    # Control gains (using similar values to CircularMotion)
    self.kp = 150.0
    self.kd = 10.0

    self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")
    self.current_idx = 0

    #as of now we can only run in a specific order
    self.order = [2, 1, 5, 0, 3, 7, 4, 6]
    #but below is the fastest order
    # self.order = list(self.best_path(self.target_points))
    self.waypoint = None

  def getIK(self, target_position, max_iters = 4):
    intr = 0

    ee_id = self.ee_id
    dx = target_position-self.d.xpos[ee_id].copy()
    jacp =  np.zeros((3, self.m.nv)) 
    jcar = np.zeros((3, self.m.nv))

    initial_jpos = np.copy(self.d.qpos[:6])
    target_jpos = np.copy(initial_jpos)

    while np.linalg.norm(dx) >= 0.01 and intr< max_iters:
      mujoco.mj_jac(self.m, self.d, jacp, jcar, self.d.xpos[self.ee_id], self.ee_id)
      EE_pos = self.d.body(ee_id).xpos
      J = jacp.copy()
      dx = target_position-EE_pos
      
      dq = J.T @ np.linalg.inv(J@J.T+0.01**2*np.eye(3))@dx

      target_jpos += 0.1*dq

      self.d.qpos[:6] = target_jpos 
      mujoco.mj_kinematics(self.m, self.d)
      
      intr+=1

    self.d.qpos[:6] = np.copy(initial_jpos)
    jpos_error = target_jpos - self.d.qpos[:6]
    velocity = self.d.qvel[:6]
    return jpos_error, velocity
  
  def best_path(self, target_points):
    ee_pos = self.d.xpos[self.ee_id].copy()
    a = [0,1,2,3,4,5,6,7]
    total_dis = float('inf')
    fast_order = []

    dist_between = np.zeros((8,8))
    for j in range(8):
      for i in range(8):
        dist_between[j,i] = np.linalg.norm(target_points[:,j] - target_points[:,i])

    for p in permutations(a):
      temp_dis = np.linalg.norm(ee_pos - target_points[:,p[0]])
      for i in range(7):
        temp_dis+=dist_between[p[i], p[i+1]]
      if temp_dis<total_dis:
        fast_order = p
        total_dis = temp_dis

    return fast_order

  def CtrlUpdate(self):
    
    index = self.order[0]
    target_position_final = self.target_points[:, index].copy()
    ee_pos = self.d.xpos[self.ee_id].copy()
    
    distance_to_final = np.linalg.norm(ee_pos - target_position_final)

    current_target = target_position_final.copy() 
    
    if self.waypoint is None:
      if distance_to_final > 0.4 and index in [6]:
        mid_point = (ee_pos + target_position_final) / 2
        
        self.waypoint = mid_point
        self.waypoint[2] += 0.3
        
        #print(f"Set waypoint: {self.waypoint}")

    if self.waypoint is not None:
      distance_to_waypoint = np.linalg.norm(ee_pos - self.waypoint)
      
      if distance_to_waypoint < 0.01:
        #print(f"Reached waypoint. Moving to final target.")
        self.waypoint = None
        current_target = target_position_final.copy()
      else:
        current_target = self.waypoint.copy()

    if distance_to_final < 0.01:
        self.order.pop(0)
        self.waypoint = None
        #print(f"Reached final target: {index}. Remaining: {self.order}")

    max_iter = 5

    jpos_error, velocity = self.getIK(current_target, max_iter)

    M = np.zeros((6,6))
    mujoco.mj_fullM(self.m, M, self.d.qM)  

    #TODO:Use Operational Space Dynamics from HW6 and lecturue 22
    jtorque_cmd = M@(self.kp*(jpos_error) - self.kd*velocity)+ self.d.qfrc_bias

    return jtorque_cmd