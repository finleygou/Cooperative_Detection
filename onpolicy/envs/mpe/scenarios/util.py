import numpy as np
from scipy.optimize import linprog
import scipy.optimize
import copy
from onpolicy.envs.mpe.intercept_probability import compute_area
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
  
# # other util functions
def Get_antiClockAngle(v1, v2):  # 向量v1逆时针转到v2所需角度。范围：0-2pi
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    assert TheNorm!=0.0, "0 in denominator"
    # 叉乘
    rho = np.arcsin(np.cross(v1, v2)/TheNorm)
    # 点乘
    cos_ = np.dot(v1, v2)/TheNorm
    if 1.0 < cos_: 
        cos_ = 1.0
        rho = 0
    elif cos_ < -1.0: 
        cos_ = -1.0
    theta = np.arccos(cos_)
    if rho < 0:
        return np.pi*2 - theta
    else:
        return theta

def Get_Beta(v1, v2):  
    # 规定逆时针旋转为正方向，计算v1转到v2夹角, -pi~pi
    # v2可能为0向量
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        # print('0 in denominator ')
        cos_ = 1  # 初始化速度为0，会出现分母为零
        return np.arccos(cos_)  # 0°
    else: 
        TheNorm = norm1*norm2
        # 叉乘
        rho = np.arcsin(np.cross(v1, v2)/TheNorm)
        # 点乘
        cos_ = np.dot(v1, v2)/TheNorm
        if 1.0 < cos_: 
            cos_ = 1.0
            rho = 0
        elif cos_ < -1.0: 
            cos_ = -1.0
        theta = np.arccos(cos_)
        if rho < 0:
            return -theta
        else:
            return theta

def GetAcuteAngle(v1, v2):  # 计算较小夹角(0-pi)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        # print('0 in denominator ')
        cos_ = 1  # 初始化速度为0，会出现分母为零
    else:  
        cos_ = np.dot(v1, v2)/(norm1*norm2)
        if 1.0 < cos_: 
            cos_ = 1.0
        elif cos_ < -1.0: 
            cos_ = -1.0
    return np.arccos(cos_)

def rad2deg(rad):
    return rad/np.pi*180

def deg2rad(deg):
    return deg/180*np.pi

def is_left(v1, v2):
    # 判断v2是否在v1的左边
    return np.cross(v1, v2) > 0

def constrain_angle(theta):
    # constrain theta in [0, 2 pi]
    if theta > 2*np.pi:
        theta = theta - np.pi*2
    elif theta < 0.0:
        theta = theta + np.pi*2
    return theta

def rand_assign_targets(num_target, num_attacker):
    '''
    return a list of target index for attackers
    '''
    if num_attacker < num_target:
        # 随机移除num_target-num_attacker个target,剩下完全匹配
        target_index = list(range(num_target))
        np.random.shuffle(target_index)
        target_index = target_index[:num_attacker]
        return target_index
    elif num_attacker == num_target:
        # 完全匹配
        target_index = list(range(num_target))
        np.random.shuffle(target_index)
        return target_index
    else:
        # 先为num_target个attackers完全分配target，剩下的attackers随机分配
        attacker_index = list(range(num_attacker))
        np.random.shuffle(attacker_index)
        target_index = np.zeros(num_attacker, dtype=int)
        for i in range(num_target):
            target_index[attacker_index[i]] = i
        for i in range(num_target, num_attacker):
            target_index[attacker_index[i]] = np.random.choice(num_target)
        return target_index
    # list_ = rand_assign_targets(6, 3)
    # print(list_)

def target_assign(T):
    '''
    task allocation algorithm based on linear programming
    '''
    # minimize the total cost
    cost_matrix = np.array(T)

    # Flatten the cost matrix to a 1D array
    c = cost_matrix.flatten()
    # print(c)

    # Number of weapons and targets
    num_weapons = cost_matrix.shape[0]
    num_targets = cost_matrix.shape[1]

    # Constraints to ensure each target gets at least one weapon
    A = np.eye(num_targets)
    for i in range(num_weapons-1):
        A = np.hstack([A, np.eye(num_targets)])
    b = np.ones(num_targets)

    # Constraints to ensure each weapon gets a target
    A_eq = np.zeros((num_weapons, num_weapons * num_targets))

    # Constraints for targets
    for i in range(num_weapons):
        for j in range(num_targets*num_weapons):
            if j == i*num_targets:
                A_eq[i, j:j+num_targets] = 1
                break

    # Right-hand side of the constraints
    b_eq = np.ones(num_weapons)

    # Bounds for each variable (0 or 1)
    x_bounds = [(0, 1) for _ in range(num_weapons * num_targets)]

    # Solve the integer linear programming problem
    result = linprog(c, -A, -b, A_eq, b_eq, bounds=x_bounds)

    
    # Extract the solution
    solution = result.x.reshape(num_weapons, num_targets)

    # Display the solution
    weapon_assignment = np.where(solution > 0.5, 1, 0)
    # print("Weapon Assignment Matrix:")
    # print(weapon_assignment)

    return weapon_assignment

def get_init_cost(attacker, defender, target):
    '''
    based on dist
    '''
    cost = np.linalg.norm(attacker.state.p_pos-target.state.p_pos) + np.linalg.norm(defender.state.p_pos-attacker.state.p_pos)
    return cost

def get_energy_cost(attacker, defender, target):
    '''
    cost based on energy, the smaller the better
    '''
    attacker_ = copy.deepcopy(attacker)
    defender_ = copy.deepcopy(defender)
    target_ = copy.deepcopy(target)
    dist_coeff = 0.01 # tunable
    cost = 0
    dt = 0.1
    t = 0
    # 模拟未来的步数
    while t<3:
        if np.linalg.norm(attacker_.state.p_vel)<0.001:  # D命中A
            cost -= 5
            break
        if np.linalg.norm(target_.state.p_vel)<0.001 or np.linalg.norm(defender_.state.p_vel)<0.001:
            break
        attacker_act = attacker_.action_callback(target_, attacker_)
        denefder_act = defender_.action_callback(target_, attacker_, defender_)
        target_act = target_.action_callback(target_, attacker_, defender_)
        cost += np.sum(np.square(denefder_act))+np.sum(np.square(target_act))+dist_coeff*np.linalg.norm(attacker_.state.p_pos-defender_.state.p_pos)
        va = attacker_.state.p_vel + attacker_act * dt
        vt = target_.state.p_vel + target_act * dt
        vd = defender_.state.p_vel + denefder_act * dt
        attacker_.state.p_vel = va / np.linalg.norm(va) * attacker_.max_speed
        target_.state.p_vel = vt / np.linalg.norm(vt) * target_.max_speed
        defender_.state.p_vel = vd / np.linalg.norm(vd) * defender_.max_speed
        attacker_.state.p_pos += attacker_.state.p_vel * dt
        target_.state.p_pos += target_.state.p_vel * dt
        defender_.state.p_pos += defender_.state.p_vel * dt
        t += dt

    del attacker_
    del defender_
    del target_

    return cost

def get_dist_cost(attacker, defender, target):
    '''
    cost based on distance and theta
    '''
    dist_coeff = 0.01 # tunable
    LOS_coeff = 0.5
    x_da = attacker.state.p_pos - defender.state.p_pos
    v_d = defender.state.p_vel
    theta_da_los = GetAcuteAngle(x_da, v_d)
    cost = LOS_coeff * theta_da_los + dist_coeff * np.linalg.norm(x_da)
    
    return cost

def get_coverage_cost_AT(attacker, target):
    '''
    cost based on distance and coverage area, for one target ad one attacker
    '''
    attacker_ = copy.deepcopy(attacker)
    target_ = copy.deepcopy(target)
    attacker_.detect_phi = attacker_.get_init_detect_direction(target_.state.p_pos-attacker_.state.p_pos)

    # compute the coverage area of the target and attacker, similar to the intercept_probability function
    tar_poly = target_.polygon_area
    bound = [(-attacker_.detect_range, attacker_.detect_range)]
    res = scipy.optimize.minimize(compute_area, attacker_.detect_phi, args=(tar_poly, [attacker_]), 
                            method='Nelder-Mead', bounds=bound)

    attacker_.detect_phi = res.x
    attacker_.detect_area = attacker_.get_detect_area()
    intersection_ = tar_poly.intersection(attacker_.detect_area)
    if intersection_.is_empty:
        attacker_.detect_phi = attacker_.get_init_detect_direction(target.state.p_pos-attacker_.state.p_pos)
        coverage_cost = 0
    else:
        coverage_cost = intersection_.area/tar_poly.area 

    dist_coeff = 0.01 # tunable
    LOS_coeff = 0.5
    x_ta = attacker.state.p_pos - target.state.p_pos
    v_t = target.state.p_vel
    theta_da_los = GetAcuteAngle(x_ta, v_t)
    cost = LOS_coeff * theta_da_los + dist_coeff * np.linalg.norm(x_ta)
    
    return cost