import mujoco
import numpy as np


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

  def getIK(self, target_position):
    #TODO: add orientation error caculation, keep current self.d.xquat[self.ee_id].copy()
    intr = 0

    point = np.zeros(3)
    ee_id = self.ee_id
    dx = target_position-self.d.xpos[ee_id].copy()
    jacp =  np.zeros((3, self.m.nv)) 
    jcar = np.zeros((3, self.m.nv))

    initial_jpos = np.copy(self.d.qpos[:6])
    target_jpos = np.copy(initial_jpos)

    while np.linalg.norm(dx) >= 0.01 and intr<3:
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
    #determine either the best path or the next best node to get
    pass

  def CtrlUpdate(self):
    
    #call best_path
    target_position = self.target_points[:, self.current_idx]

    ee_pos = self.d.xpos[self.ee_id].copy()
    distance = np.linalg.norm(ee_pos - target_position)
    if distance < 0.01:
      #call bestpath?
      self.current_idx+=1
      target_position = self.target_points[:, self.current_idx]
      if self.current_idx == 8:
        return np.zeros((6,6))

    jpos_error, velocity = self.getIK(target_position)

    M = np.zeros((6,6))
    mujoco.mj_fullM(self.m, M, self.d.qM)  

    #TODO:Use Operational Space Dynamics from HW6 and lecturue 22
    jtorque_cmd = M@(self.kp*(jpos_error) - self.kd*velocity)+ self.d.qfrc_bias

    return jtorque_cmd