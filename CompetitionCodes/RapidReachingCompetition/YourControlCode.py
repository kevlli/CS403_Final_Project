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

  def getIK(self, goal, initq):
    #TODO: add orientation error caculation, keep current dataT.xquat[self.ee_id].copy()
    dataT = mujoco.MjData(self.m)
    intr = 0
    dataT.qpos[:] = initq.copy() 
    mujoco.mj_forward(self.m, dataT)    

    dx = goal-dataT.xpos[self.ee_id].copy()

    while np.linalg.norm(dx) > 0.01 and intr<100:
      jacp =  np.zeros((3, self.m.nv)) 
      jcar = np.zeros((3, self.m.nv))
      mujoco.mj_jac(self.m, dataT, jacp, jcar, dataT.xpos[self.ee_id], self.ee_id)
      J = jacp.copy()
      
      dq = J.T @ np.linalg.inv(J@J.T+0.01**2*np.eye(3))@dx

      dataT.qpos[:] = dataT.qpos[:]+0.01*dq

      mujoco.mj_forward(self.m, dataT)
      
      dx = goal-dataT.xpos[self.ee_id].copy()
      intr+=1
    return dataT.qpos[:]
  
  def best_path(self, target_points):
    #determine either the best path or the next best node to get
    pass

  def CtrlUpdate(self):
    
    #call best_path
    goal = self.target_points[:, self.current_idx]

    ee_pos = self.d.xpos[self.ee_id].copy()
    distance = np.linalg.norm(ee_pos - goal)
    if distance < 0.01:
      #call bestpath?
      self.current_idx+=1
      goal = self.target_points[:, self.current_idx]

    q_d = self.getIK(goal, self.d.qpos)

    M = np.zeros((6,6))
    mujoco.mj_fullM(self.m, M, self.d.qM)  

    #TODO:Use Operational Space Dynamics from HW6 and lecturue 22
    jtorque_cmd = M@(self.kp*(q_d - self.d.qpos) - self.kd*self.d.qvel)+ self.d.qfrc_bias

    return jtorque_cmd