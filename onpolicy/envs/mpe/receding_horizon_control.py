from .intercept_probability import *

'''
手动实现多智能体的MPC控制
没有调用python的MPC包是因为其不利于处理多智能体任务
'''

def system_dynamics(agent, dt, u):
    v = agent.state.p_vel + u * dt
    if np.linalg.norm(v) != 0:
        v = v / np.linalg.norm(v) * agent.max_speed
    # print("{} {} done is {}".format(agent.name, agent.id, agent.done))
    if agent.done:
        agent.state.p_vel = np.array([0, 0]) # 速度清零
    else:
        v_x, v_y = v[0], v[1]
        theta = np.arctan2(v_y, v_x)
        if theta < 0:
            theta += np.pi * 2
        # update phi
        agent.state.phi = theta
        agent.state.p_vel = np.array([v_x, v_y])
    agent.state.p_pos += agent.state.p_vel * dt

    return agent

def cost_function(u, *args):
    a1 = 100.  # 截获概率的权重
    a2  = 10.  # 控制能量的权重

    # u only contains the control of attackers
    target, attackers, dt = args[0], args[1], args[2]
    
    # 深拷贝对象的深拷贝
    target_ = copy.deepcopy(target)
    attackers_ = copy.deepcopy(attackers)

    # print("u: ", u[0:4])
    u = np.array(u)
    u = u.reshape((-1, len(attackers_)))
    h = u.shape[0]

    
    cost = 0
    for i in range(h):
        # forward dynamics
        u_i = u[i, :]
        for j, att in enumerate(attackers_):
            e_v = att.state.p_vel/np.linalg.norm(att.state.p_vel)
            e_am = np.array([- e_v[1], e_v[0]])
            u_ij = np.multiply(e_am, u_i[j])  
            attackers_[j] = system_dynamics(att, dt, u_ij)

        target_u = target_.action_callback(target_, target_.mode, 0)
        target_ = system_dynamics(target_, dt, target_u)

        # compute cost
        sum_u = np.sum(u_i**2)
        target_poly = target_.polygon_area
        # opt_detect = detection_optimization(target_, attackers_)
        opt_detect = []
        for agent in attackers_:
            agent.detect_phi = agent.get_init_detect_direction(target_.state.p_pos-agent.state.p_pos)
            agent.detect_area = agent.get_detect_area()
            opt_detect.append(agent.detect_phi)
        probability = compute_area(opt_detect, target_poly, attackers_)
        cost += a1 * probability/target_.area + a2 * sum_u

    del target_
    del attackers_

    # print("cost: ", cost)

    return cost

def receding_horizon(target, attackers, dt):
    # print("in receding horizon control")

    h = 5  # 预测步长
    N = len(attackers)
    u0 = np.zeros(h*N)  # 控制量, 拉成一行便于优化. u0 = [u1(1), u2(1), ..., uh]

    target_ = copy.deepcopy(target)
    attackers_ = copy.deepcopy(attackers)

    # 预处理u0, 添加约束
    bound0 = []
    for i, att in enumerate(attackers_):
        e_v = att.state.p_vel/np.linalg.norm(att.state.p_vel)
        e_am = np.array([- e_v[1], e_v[0]])  # ，逆旋90°，垂直于速度方向的单位向量，u正方向
        u_i = np.dot(att.action.u, e_am)  # 实数 标量
        u0[i] = u_i
        bound0.append((-att.max_accel, att.max_accel))
    
    bound = bound0 * h

    # 优化
    res = scipy.optimize.minimize(cost_function, u0, args=(target_, attackers_, dt), 
                            method='SLSQP', bounds=bound)
    
    u1 = res.x
    u1 = np.array(u1[0:N])

    # print("u0: ", u0[0:N])
    # print("u1: ", u1)

    # 后处理u
    u = []
    for i, att in enumerate(attackers_):
        e_v = att.state.p_vel/np.linalg.norm(att.state.p_vel)
        e_am = np.array([- e_v[1], e_v[0]])
        u_i = np.multiply(u1[i], e_am)
        u.append(u_i)

    del target_
    del attackers_

    return u

    




