import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Target, Attacker
from onpolicy.envs.mpe.scenario import BaseScenario
from onpolicy import global_var as glv
import sys
from util import *
from shapely.geometry import Point

'''
距离单位: km
时间单位: s
'''
Mach = 0.34  # km/s
G = 9.8e-3  # km/s^2

global a1, b1, c1, d1, N_m
a1 = 0
b1 = 1e10
c1 = 0.5
d1 = 0.5
N_m = 4

class Scenario(BaseScenario):
    
    def __init__(self) -> None:
        super().__init__()
        self.cp = 0.4
        self.use_CL = 0  # 是否使用课程式训练(render时改为false)
        self.assign_list = [] # 为attackers初始化分配target, 长度为num_A, 值为target的id
        # self.init_assign = True  # 仅在第一次分配时使用

    # 设置agent,landmark的数量，运动属性。
    def make_world(self,args):
        self.num_target = args.num_target
        self.num_attacker = args.num_attacker

        world = World()
        world.collaborative = True
        world.world_length = args.episode_length
        # set any world properties first
        # add agents
        world.targets = [Target() for i in range(self.num_target)]  # 4
        world.attackers = [Attacker() for i in range(self.num_attacker)]  # 10
        world.agents = world.targets+world.attackers

        for i, target in enumerate(world.targets):
            target.id = i
            target.size = 0.1
            target.color = np.array([0.15, 0.95, 0.15]) #green
            target.color_true_pos = np.array([0.90, 0.60, 0.20]) #orange
            target.max_speed = 1.0*Mach
            target.max_accel = 5*G
            target.action_callback = target_policy

        for i, attacker in enumerate(world.attackers):
            attacker.id = i
            attacker.size = 0.1
            attacker.color = np.array([0.95, 0.15, 0.15])
            attacker.max_speed = 4*Mach
            attacker.max_accel = 15*G
            attacker.action.u = np.zeros(2)
            attacker.action_callback = attacker_policy

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        self.init_assign = True
        self.assign_list = rand_assign_targets(self.num_target, self.num_attacker)
        # print('init assign_list is:', self.assign_list)

        # properties and initial states for agents
        init_pos_target = np.array([[24.0, 3.0], [20, 14.0], [22, -5.0], [21, -11.5]])
        init_pos_target = init_pos_target + np.random.randn(*init_pos_target.shape)*0.2
        for i, target in enumerate(world.targets):
            target.done = False
            target.state.p_pos = init_pos_target[i]
            target.state.p_vel = np.array([target.max_speed, 0.0])
            target.state.V = np.linalg.norm(target.state.p_vel)
            target.state.phi = 0.
            target.attacker = [j for j in range(self.num_attacker) if self.assign_list[j]==i]  # the id of attacker in world.attackers
            target.defender = None
            target.attackers = []
            target.defenders = []
            target.cost = []

            target.sigma_dist = 4  # 区域半径
            rand_pos = np.random.uniform(0, 1, 2)
            r_, theta_ = target.sigma_dist*rand_pos[0], np.pi*2*rand_pos[1]
            target.state.p_pos_true = target.state.p_pos + np.array([r_*np.cos(theta_), r_*np.sin(theta_)])
            # target.state.p_pos_true = target.state.p_pos + np.array([-0.4, -0.8])*target.sigma_dist
            target.area_mode = 4  # 1:长方形  2:5边形  3:6边形  4:圆形
            target.polygon_area = target.get_area()


        init_pos_attacker = np.array([[-1.2, 10.5], [2.0, 7.5], [3.0, 5.0], [0.3, 3.0], [-0.5, 0.4],
                                      [1.0, -3.5], [3.5, -5.0], [0.0, -7.0], [-0.2, -9.5], [2.5, -13.5]])
        init_pos_attacker = init_pos_attacker + np.random.randn(*init_pos_attacker.shape)*0.1
        for i, attacker in enumerate(world.attackers):
            attacker.done = False
            attacker.state.p_pos = init_pos_attacker[i]
            attacker.state.p_vel = np.array([attacker.max_speed, 0.0])
            attacker.state.V = np.linalg.norm(attacker.state.p_vel)
            attacker.state.phi = 0.
            attacker.true_target = self.assign_list[i]
            attacker.fake_target = self.assign_list[i]
            attacker.last_belief = attacker.fake_target
            attacker.flag_kill = False
            attacker.flag_dead = False
            attacker.is_locked = False
            # attacker.defenders = []

            attacker.detect_phi = attacker.get_init_detect_direction(world.targets[0].state.p_pos-attacker.state.p_pos)
            attacker.detect_area = attacker.get_detect_area()

        self.update_belief(world)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        if agent2.name == 'target':
            dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos_true)
        elif agent1.name == 'target':
            dist = np.linalg.norm(agent1.state.p_pos_true - agent2.state.p_pos)
        else:
            dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def target_defender(self, world):
        return [agent for agent in world.agents if not agent.name=='attacker']

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def set_CL(self, CL_ratio):
        d_cap = 1.5
        if CL_ratio < self.cp:
            # print('in here Cd')
            self.d_cap = d_cap*(self.cd + (1-self.cd)*CL_ratio/self.cp)
        else:
            self.d_cap = d_cap

    # agent 和 adversary 分别的reward
    def reward(self, agent, world):
        if agent.name == 'attacker':
            main_reward = self.attacker_reward(agent, world)  # else self.agent_reward(agent, world)
        else:
            print('reward error')
        return main_reward

    # individual adversary award
    def attacker_reward(self, attacker, world):  # agent here is adversary
        '''
        r_k : reward for killing target or being killed
        r_d : reward for deception situation
        r_s : reward for switch in belief (actually penalty)
        r_p : penalty for choosing a dead fake target
        '''
        # r_k, r_s, r_p, r_d = 0, 0, 0, 0

        # # 好的奖励持续给，坏奖励只给一次
        # # reward kill
        # if attacker.flag_kill:
        #     # attacker.flag_kill = False
        #     r_k = 10
        # # penalty being killed
        # if attacker.flag_dead:
        #     attacker.flag_dead = False
        #     r_k = -8

        # if not attacker.done:
        #     # switch in belief
        #     if attacker.last_belief != attacker.fake_target:
        #         r_s = -1
            
        #     # prevent choosing dead target
        #     if world.targets[attacker.fake_target].done:
        #         r_p = -2

        #     # deception reward using apollonius circle
        #     target = world.targets[attacker.true_target]
        #     alpha = target.state.V / attacker.state.V
        #     r1, r2, r3, r4 = 3, 1, 0, -1
        #     for i, defender_id in enumerate(attacker.defenders):
        #         defender = world.defenders[defender_id]
        #         beta = defender.state.V / attacker.state.V
        #         x_ad = defender.state.p_pos - attacker.state.p_pos
        #         r_ = np.linalg.norm(x_ad)
        #         e_ad = x_ad / r_
        #         x_at = target.state.p_pos - attacker.state.p_pos
        #         R_ = np.linalg.norm(x_at)
        #         e_at = x_at / R_
        #         cos_ = np.dot(e_at, e_ad)
        #         cos_Theta = (r_*r_ + R_*R_ - (alpha*r_+beta*R_)**2)/(2*r_*R_)
        #         cos_eta_gamma = np.sqrt((1-alpha**2)*(1-beta**2))-alpha*beta
        #         if -1 <= cos_ <cos_eta_gamma:
        #             r_p += r2 + (r1-r2)/(-1-cos_eta_gamma)*(cos_ - cos_eta_gamma)
        #         elif cos_eta_gamma <= cos_ < cos_Theta:
        #             r_p += r2 + (r3-r2)/(cos_Theta-cos_eta_gamma)*(cos_ - cos_eta_gamma)
        #         else:
        #             r_p += r3 + (r4-r3)/(1-cos_Theta)*(cos_ - cos_Theta)

        # rew = r_k + r_s + r_p + r_d
        rew = 1

        return rew


    # observation for adversary agents
    def observation(self, agent, world):
        # o_ego  1+2+2=5
        return np.array([1])

    def done(self, agent, world):  # 
        # for attackers only
        if self.is_collision(agent, world.targets[agent.true_target]) and not agent.done:
            agent.done = True 
            agent.flag_kill = True
            world.targets[agent.true_target].done = True
            return True
        
        
        # 要加下面这句话，否则done一次后前面两个if不会进入，立刻又变成False了。
        if agent.done:
            return True
        
        return False

    def update_belief(self, world):
        '''
        update targets' belief according to the assignment result, which attackers are assigned to which targets
        '''
        target_list = []
        defender_list = []
        attacker_list = []
        target_id_list = []
        defender_id_list = []
        attacker_id_list = []  
        for target in world.targets:
            if not target.done:
                target_list.append(target)
                target_id_list.append(target.id)
                target.defenders = []
                target.attackers = [] # 重新分配ADs
                target.cost = []
        for defender in world.defenders:
            if not defender.done:
                defender_list.append(defender)
                defender_id_list.append(defender.id)
        for attacker in world.attackers:
            if not attacker.done:
                attacker.defenders = []
                attacker_list.append(attacker)
                attacker_id_list.append(attacker.id)
        
        if len(target_list) > 0:
            # calculate the cost matrix
            T = np.zeros((len(attacker_list), len(target_list)))
            if self.init_assign:
                for i, attacker in enumerate(attacker_list):
                    for j, target in enumerate(target_list):
                        T[i, j] = get_coverage_cost_AT(attacker, target)
                assign_result = target_assign(T)  # 线性规划，初次分配解
                self.init_assign = False
            else: 
                # 非线性整数规划，最大化探测面积
                # assign_result = target_assign_NonlinearInteger(T, attacker_list, target_list)  # |A|*|T|的矩阵
            
                for i, attacker in enumerate(attacker_list):
                    for j, target in enumerate(target_list):
                        T[i, j] = get_coverage_cost_AT(attacker, target)
                assign_result = target_assign(T)  # 线性规划，初次分配解
            # print('assign_result is:', assign_result)

            '''
            如果assign报错，检查'TAD list，查看A的list是否为空。如果全部A被拦截，不需要update belief
            '''
            # update belief list of TDs according to assign_result
            for i, attacker in enumerate(attacker_list):  # 遍历行
                for j in range(len(target_list)):
                    if assign_result[i, j] == 1:
                        attacker.fake_target = target_list[j].id
                        attacker.true_target = target_list[j].id
                        target = target_list[j]
                        target.attackers.append(attacker.id)
                        # target.cost.append(T[i, j])
            
            # 为target从list中选择AD
            for i, target in enumerate(target_list):
                if len(target.cost)>0:
                    # target.defender = target.defenders[np.argmin(target.cost)]
                    target.attacker = target.attackers[np.argmin(target.cost)]
                else:
                    # 有的target已经不需要A了，A太少了
                    # target.defender = np.random.choice(defender_id_list)
                    target.attacker = np.random.choice(attacker_id_list)
            # print('T believes are:', )

    def check_found_target(self, target, world):

        target_true_pos = Point(target.state.p_pos_true)
        is_detected = False
        for i in range(len(target.attackers)):
            attacker = world.attackers[target.attackers[i]]
            if attacker.detect_area.contains(target_true_pos):
                is_detected = True
        
        if is_detected:
            target.state.p_pos = target.state.p_pos_true
            for i in range(len(target.attackers)):
                attacker = world.attackers[target.attackers[i]]
                attacker.is_locked = True
                attacker.true_target = target.id

        return is_detected


'''
low-level policy for TADs
'''
def target_policy(target, mode, t):
    if target.done:
        return np.array([0, 0])
    
    if mode == 1:
        # 直线运动
        return np.array([0, 0])

    elif mode == 2:
        # S型运动
        return np.array([np.sin(t), np.cos(t)])

    elif mode == 3:
        # 圆形运动
        r_ = 5  # 圆的半径
        a_norm = target.state.V**2 / r_
        if a_norm > target.max_accel:
            a_norm = target.max_accel
        e_vt = target.state.p_vel / np.linalg.norm(target.state.p_vel)
        e_uq = np.array([- e_vt[1], e_vt[0]])
        a = a_norm * e_uq

        return a

'''
def defender_policy(target, attacker, defender):
    global a1, b1, c1, d1, N_m

    if target.done or attacker.done or defender.done:
        return np.array([0, 0])

    # 期望攻击角度
    q_md_expect = 0.

    # 各方状态
    V_m = attacker.state.p_vel
    V_t = target.state.p_vel
    V_d = defender.state.p_vel
    e_vm = V_m / np.linalg.norm(V_m)
    e_vt = V_t / np.linalg.norm(V_t)
    e_vd = V_d / np.linalg.norm(V_d)
    x_mt = target.state.p_pos - attacker.state.p_pos
    r_mt = np.linalg.norm(x_mt)
    e_mt = x_mt / r_mt
    x_md = defender.state.p_pos - attacker.state.p_pos
    r_md = np.linalg.norm(x_md)
    e_md = x_md / r_md
    q_md = np.arctan2(x_md[1], x_md[0])
    q_md = q_md + np.pi * 2 if q_md < 0 else q_md  # 0~2pi

    # TAD基本关系
    r_mt_dot = np.dot(V_t - V_m, e_mt)
    q_mt_dot = np.cross(e_mt, V_t - V_m) / r_mt

    r_md_dot = np.dot(V_d - V_m, e_md) # negative
    q_md_dot = np.cross(e_md, V_d - V_m) / r_md

    # 旧的状态量
    x_11 = q_md - q_md_expect
    x_12 = q_md_dot

    # 中间参数
    M1 = N_m * np.dot(e_md, e_vm) / (5 * np.dot(e_mt, e_vm))
    P1 = N_m * np.dot(e_md, e_vm) * r_mt_dot * q_mt_dot / (np.dot(e_mt, e_vm) * r_md)
    L1 = 1 / c1 + M1 * M1 / d1
    A1 = L1 * a1 / (8 * r_md_dot * r_md_dot) * (1 - 2 * np.e * np.e + np.power(np.e, 4))
    B1 = L1 * b1 / (4 * r_md * r_md_dot) * (1 - np.power(np.e, 4)) + 1
    C1 = np.e * np.e * x_12 - r_md * P1 * (1 - np.e * np.e) / (2 * r_md_dot)
    D1 = L1 * a1 * r_md / (16 * np.power(r_md_dot, 3)) * (4 + np.power(np.e, -2) - np.e * np.e) - 1
    E1 = r_md / (2 * r_md_dot) * (np.power(np.e, -2) - 1) + L1 * b1 / (8 * r_md_dot * r_md_dot) * (np.e * np.e + np.power(np.e, -2) - 2)
    F1 = - x_11 - r_md * r_md * P1 * (1 + np.power(np.e, -2)) / (4 * r_md_dot * r_md_dot)

    # 新的状态量
    x_11_tf = (B1 * F1 - C1 * E1) / (B1 * D1 - A1 * E1)
    x_12_tf = (C1 * D1 - A1 * F1) / (B1 * D1 - A1 * E1)

    # 协态量
    # lamda_11 = a1 * x_11_tf
    lamda_12 = a1 * x_11_tf * r_md / (2 * r_md_dot) * (1 - np.e * np.e) + b1 * x_12_tf * np.e * np.e

    # 加速度
    w_q = lamda_12 / (c1 * r_md)
    e_wq = - np.array([- e_md[1], e_md[0]])
    if np.linalg.norm(V_d) == 0:
        a_d = np.array([0, 0])
    else:
        e_ad = np.array([- e_vd[1], e_vd[0]])
        if GetAcuteAngle(e_ad, e_wq) > np.pi / 2:
            e_ad = np.array([e_ad[1], - e_ad[0]])
        a_d_value = np.clip(w_q / np.abs(np.dot(-e_md, e_vd)), - defender.max_accel, defender.max_accel)
        a_d = np.multiply(e_ad, a_d_value)

    # a_d = np.array([0, 0])

    return a_d
'''

def attacker_policy(target, attacker):
    global N_m
    # target = world.targets[attacker.fake_target]

    if target.done or attacker.done:
        return np.array([0, 0])

    # TAD基本关系
    # 要有向量的思想，起点和终点
    V_m = attacker.state.p_vel
    V_t = target.state.p_vel
    x_mt = target.state.p_pos - attacker.state.p_pos
    r_mt = np.linalg.norm(x_mt)
    e_mt = x_mt / r_mt
    r_mt_dot = np.dot(V_t - V_m, e_mt)
    q_mt_dot = np.cross(e_mt, V_t - V_m) / r_mt

    # attacker的加速度
    u_q = - N_m * r_mt_dot * q_mt_dot # PNG
    # u_q = - N_m * r_mt_dot * q_mt_dot - 0.2 * N_m * target.state.controller # APNG
    '''
    APNG效果不好, target的动作会导致attacker的动作不稳定
    '''
    e_uq = np.array([- e_mt[1], e_mt[0]])  # 单位方向向量
    if np.linalg.norm(V_m) == 0:
        a_m = np.array([0, 0])
    else:
        e_v = V_m / np.linalg.norm(V_m)
        e_am = np.array([- e_v[1], e_v[0]])  # v方向逆转90°，单位方向向量
        if GetAcuteAngle(e_am, e_uq) > np.pi / 2:
            e_am = np.array([e_v[1], - e_v[0]])
        a_m_value = np.clip(u_q / np.abs(np.dot(e_uq, e_am)), - attacker.max_accel, attacker.max_accel)
        a_m = np.multiply(e_am, a_m_value)

    return a_m

