import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon

# import gif

'''
computing area 可以换成计算截获概率的公式
也可以动态修改目标区域的权重，让已经被覆盖过的区域权重比例降低
'''

def detection_optimization(target, attackers):
    tar_poly = target.polygon_area
    attackers_ = copy.deepcopy(attackers)
    att_poly_origin = []  # pts
    att_pos = []
    att_phi = []
    att_direction = []  # 待优化变量，x0

    for att in attackers_:
        att_pos.append(att.state.p_pos)
        att_phi.append(att.state.phi)
        att_direction.append(att.detect_phi)
        att.detect_phi = att.get_init_detect_direction(target.state.p_pos-att.state.p_pos)  # initial direction, 每次优化都重新初始化
        att_poly_origin.append(list(att.get_detect_area().exterior.coords[:-1]))  # points to be visualized
    
    bound = [(-attackers_[0].detect_range, attackers_[0].detect_range)] * len(attackers_)
    # print('the bound is:', bound)
    # print('the initial direction is:', att_direction)

    res = scipy.optimize.minimize(compute_area, att_direction, args=(tar_poly, attackers_), 
                            method='SLSQP', bounds=bound)
    x = res.x

    # 检验是否有poly与目标poly相交为空集，若有则更新该x朝向target
    for i, att in enumerate(attackers_):
        att.detect_phi = x[i]
        att.detect_area = att.get_detect_area()
        intersection_ = tar_poly.intersection(att.detect_area)
        if intersection_.is_empty:
            x[i] = att.get_init_detect_direction(target.state.p_pos-att.state.p_pos)

    ##################### draw #####################
    tar_poly_pts = [list(tar_poly.exterior.coords[:-1])]
    # print('the target polygon is:', tar_poly_pts)
    all_agents = att_pos
    all_agents.append(target.state.p_pos)
    all_agents = np.array(all_agents)
    # draw(att_poly_origin, tar_poly_pts, all_agents, att_phi)

    att_poly_new = []
    for i, att in enumerate(attackers_):
        att.detect_phi = x[i]
        att.detect_area = att.get_detect_area()
        att_poly_new.append(list(att.get_detect_area().exterior.coords[:-1]))
    # draw(att_poly_new, tar_poly_pts, all_agents, att_phi)

    del attackers_

    return x

def compute_area(x, *args):
    '''
    args = (tar_poly, attackers_)
    '''
    tar_poly = args[0]
    attackers = args[1]
    att_poly = []
    num_att = len(attackers)

    for i, att in enumerate(attackers):
        att.detect_phi = x[i]
        att_poly.append(att.get_detect_area())
    
    probability = 0  # 均匀分布，概率为面积
    # 容斥原理计算面积，偶加奇减的原则

    if num_att >= 1:
        # 计算2个图形间的面积
        for i in range(len(attackers)):
            probability += compute_2area(tar_poly, att_poly[i]) 
    
    if num_att >= 2:
        # 计算3个图形间的面积
        for i in range(len(attackers)):
            for j in range(i+1, len(attackers)):
                probability -= compute_3area(tar_poly, att_poly[i], att_poly[j])
    
    if num_att >= 3:
        # 计算4个图形间的面积
        for i in range(len(attackers)):
            for j in range(i+1, len(attackers)):
                for k in range(j+1, len(attackers)):
                    probability += compute_4area(tar_poly, att_poly[i], att_poly[j], att_poly[k])
    
    if num_att >= 4:
        # 计算5个图形间的面积
        for i in range(len(attackers)):
            for j in range(i+1, len(attackers)):
                for k in range(j+1, len(attackers)):
                    for l in range(k+1, len(attackers)):
                        probability -= compute_5area(tar_poly, att_poly[i], att_poly[j], att_poly[k], att_poly[l])

    if num_att >= 5:
        # 计算6个图形间的面积
        for i in range(len(attackers)):
            for j in range(i+1, len(attackers)):
                for k in range(j+1, len(attackers)):
                    for l in range(k+1, len(attackers)):
                        for m in range(l+1, len(attackers)):
                            probability += compute_6area(tar_poly, att_poly[i], att_poly[j], att_poly[k], att_poly[l], att_poly[m])

    return probability


'''
for multi-target detection optimization scenarios
'''
# def detect_optimization_multi(targets, attackers):
#     '''
#     多目标检测优化
#     '''
#     tar_poly = []
#     for tar in targets:
#         tar_poly.append(tar.polygon_area)
#     attackers_ = copy.deepcopy(attackers)
#     att_pos = []
#     att_phi = []
#     att_direction = []  # 待优化变量，x0

#     for att in attackers_:
#         att_pos.append(att.state.p_pos)
#         att_phi.append(att.state.phi)
#         att_direction.append(att.detect_phi)
#         att.detect_phi = att.get_init_detect_direction(targets[0].state.p_pos-att.state.p_pos)  # initial direction, 每次优化都重新初始化

#     bound = [(-attackers_[0].detect_range, attackers_[0].detect_range)] * len(attackers_)
#     res = scipy.optimize.minimize(compute_area_multi, att_direction, args=(tar_poly, attackers_), 
#                             method='Nelder-Mead', bounds=bound)
#     x = res.x

#     # 检验是否有poly与目标poly相交为空集，若有则更新该x朝向target
#     for i, att in enumerate(attackers_):
#         att.detect_phi = x[i]
#         att.detect_area = att.get_detect_area()
#         for tar_poly in tar_poly:
#             intersection_ = tar_poly.intersection(att.detect_area)
#             if intersection_.is_empty:
#                 x[i] = att.get_init_detect_direction(targets[0].state.p_pos-att.state.p_pos)

#     ##################### draw #####################
#     tar_poly_pts = []
#     for tar_poly in tar_poly:
#         tar_poly_pts.append(list(tar_poly.exterior.coords[:-1]))
#     all_agents = att_pos
#     for tar in targets:
#         all_agents.append(tar.state.p_pos)
#     all_agents = np.array(all_agents)
#     # draw(att_poly_origin, tar_poly_pts, all_agents, att_phi)

#     att_poly_new = []
#     for i, att in enumerate(attackers_):
#         att.detect_phi = x[i]
#         att.detect_area = att.get_detect_area()
#         att_poly_new.append(list(att.get_detect_area().exterior.coords[:-1]))
#     # draw(att_poly_new, tar_poly_pts, all_agents, att_phi)

#     del attackers_

#     return x

# def compute_area_multi(x, *args):
#     '''
#     args = (tar_poly, attackers_)
#     '''
#     tar_poly = args[0]
#     attackers = args[1]
#     att_poly = []

#     for i, att in enumerate(attackers):
#         att.detect_phi = x[i]
#         att_poly.append(att.get_detect_area())
    
#     probability = 0  # 均匀分布，概率为面积
#     # 容斥原理计算面积，偶加奇减的原则
#     # 计算2个图形间的面积
#     for i in range(len(attackers)):
#         for j in range(len(tar_poly)):
#             probability += compute_2area(tar_poly[j], att_poly[i]) 
#     # 计算3个图形间的面积
#     for i in range(len(attackers)):
#         for j in range(i+1, len(attackers)):
#             for k in range(len(tar_poly)):
#                 probability -= compute_3area(tar_poly[k], att_poly[i], att_poly[j])
#     # 计算4个图形间的面积
#     for i in range(len(attackers)):
#         for j in range(i+1, len(attackers)):
#             for k in range(j+1, len(attackers)):
#                 for l in range(len(tar_poly)):
#                     probability += compute_4area(tar_poly[l], att_poly[i], att_poly[j], att_poly[k])
#     # 计算5个图形间的面积
#     for i in range(len(attackers)):
#         for j in range(i+1, len(attackers)):
#             for k in range(j+1, len(attackers)):
#                 for l in range(k+1, len(attackers)):
#                     for m in range(len(tar_poly)):
#                         probability -= compute_5area(tar_poly[m], att_poly[i], att_poly[j], att_poly[k], att_poly[l])

#     return probability


def compute_2area(tar_poly, att_poly1):
    intersection_ = tar_poly.intersection(att_poly1)
    if intersection_.is_empty:
        return 0
    else:
        '''
        intersection_pts = list(intersection_.exterior.coords[:-1])
        pts = np.array(intersection_pts)
        tri_cent = []
        for j in range(len(pts) - 2):
            pt1, pt2, pt3 = pts[0], pts[j + 1], pts[j + 2]
            area = 1 / 2 * np.cross(pt2 - pt1, pt3 - pt1)  # 负数
            tri_cent.append([1 / 3 * (pt1[0] + pt2[0] + pt3[0]), 1 / 3 * (pt1[1] + pt2[1] + pt3[1]), area])
        Area, sumx, sumy = 0, 0, 0
        for j in range(len(tri_cent)):
            # sumx = sumx + tri_cent[j][0] * tri_cent[j][2]
            # sumy = sumy + tri_cent[j][1] * tri_cent[j][2]
            Area = Area + tri_cent[j][2]
        # Cx = sumx / Area # 计算质心
        # Cy = sumy / Area
        '''
        Area = - intersection_.area  # 因为最后的函数是minimize，所以要取负数
        # print('Area is:', Area)
        # print('polygon area is:', intersection_.area)

        return Area

def compute_3area(tar_poly, att_poly1, att_poly2):
    intersection_1 = tar_poly.intersection(att_poly1)
    if intersection_1.is_empty:
        return 0
    else:
        intersection_1_hull = Polygon(intersection_1.exterior.coords[:-1])
        intersection_2 = intersection_1_hull.intersection(att_poly2)
        if intersection_2.is_empty:
            return 0
        else:
            # intersection_pts = list(intersection_2.exterior.coords[:-1])
            # pts = np.array(intersection_pts)
            # Area = 0
            # for j in range(len(pts) - 2):
            #     pt1, pt2, pt3 = pts[0], pts[j + 1], pts[j + 2]
            #     area = 1 / 2 * np.cross(pt2 - pt1, pt3 - pt1)
            #     Area = Area + area
            Area = - intersection_2.area
            return Area
    
def compute_4area(tar_poly, att_poly1, att_poly2, att_poly3):
    intersection_1 = tar_poly.intersection(att_poly1)
    if intersection_1.is_empty:
        return 0
    else:
        intersection_1_hull = Polygon(intersection_1.exterior.coords[:-1])
        intersection_2 = intersection_1_hull.intersection(att_poly2)
        if intersection_2.is_empty:
            return 0
        else:
            intersection_2_hull = Polygon(intersection_2.exterior.coords[:-1])
            intersection_3 = intersection_2_hull.intersection(att_poly3)
            if intersection_3.is_empty:
                return 0
            else:
                Area = - intersection_3.area
                return Area

def compute_5area(tar_poly, att_poly1, att_poly2, att_poly3, att_poly4):
    intersection_1 = tar_poly.intersection(att_poly1)
    if intersection_1.is_empty:
        return 0
    else:
        intersection_1_hull = Polygon(intersection_1.exterior.coords[:-1])
        intersection_2 = intersection_1_hull.intersection(att_poly2)
        if intersection_2.is_empty:
            return 0
        else:
            intersection_2_hull = Polygon(intersection_2.exterior.coords[:-1])
            intersection_3 = intersection_2_hull.intersection(att_poly3)
            if intersection_3.is_empty:
                return 0
            else:
                intersection_3_hull = Polygon(intersection_3.exterior.coords[:-1])
                intersection_4 = intersection_3_hull.intersection(att_poly4)
                if intersection_4.is_empty:
                    return 0
                else:
                    Area = - intersection_4.area
                    return Area

def compute_6area(tar_poly, att_poly1, att_poly2, att_poly3, att_poly4, att_poly5):
    intersection_1 = tar_poly.intersection(att_poly1)
    if intersection_1.is_empty:
        return 0
    else:
        intersection_1_hull = Polygon(intersection_1.exterior.coords[:-1])
        intersection_2 = intersection_1_hull.intersection(att_poly2)
        if intersection_2.is_empty:
            return 0
        else:
            intersection_2_hull = Polygon(intersection_2.exterior.coords[:-1])
            intersection_3 = intersection_2_hull.intersection(att_poly3)
            if intersection_3.is_empty:
                return 0
            else:
                intersection_3_hull = Polygon(intersection_3.exterior.coords[:-1])
                intersection_4 = intersection_3_hull.intersection(att_poly4)
                if intersection_4.is_empty:
                    return 0
                else:
                    intersection_4_hull = Polygon(intersection_4.exterior.coords[:-1])
                    intersection_5 = intersection_4_hull.intersection(att_poly5)
                    if intersection_5.is_empty:
                        return 0
                    else:
                        Area = - intersection_5.area
                        return Area

# @gif.frame
def draw(poly_att, poly_tar, all_agents, att_phi):
    # 绘制维诺图
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    # 母点
    ax.scatter(all_agents[:, 0], all_agents[:, 1], s=100)
    # 维诺图区域
    poly_vor = PolyCollection(poly_att, edgecolor="red",
                              facecolors="None", linewidth=1.0)
    ax.add_collection(poly_vor)

    poly_tar_ = PolyCollection(poly_tar, edgecolor="green",
                                facecolors="None", linewidth=1.0)
    ax.add_collection(poly_tar_)

    for i in range(len(att_phi)):
        x, y = all_agents[i]
        phi = att_phi[i]
        ax.quiver(x, y, 3*np.cos(phi), 3*np.sin(phi), angles='xy', scale_units='xy', scale=1, color='blue')

    # for j in range(len(obs)):
    #     x_, y_, R_, delta_ = obs[j, 0], obs[j, 1], obs[j, 2], obs[j, 3]
    #     c1 = Circle(xy=(x_, y_), radius=R_, alpha=0.5, color='grey')
    #     c2 = Circle(xy=(x_, y_), radius=R_+delta_, alpha=0.5, color='grey', fill=False, linestyle='--')
    #     ax.add_patch(c1)
    #     ax.add_patch(c2)

    # for i in range(len(target_pts)):
    #     Cx = target_pts[i][0]
    #     Cy = target_pts[i][1]
    #     plt.plot(Cx, Cy, marker='+', color='coral')

    # xmin = np.min(bound[:, 0])
    # xmax = np.max(bound[:, 0])
    # ymin = np.min(bound[:, 1])
    # ymax = np.max(bound[:, 1])
    xmin, xmax, ymin, ymax = -5, 30, -15, 15
    ax.set_xlim(xmin - 0.1, xmax + 0.1)
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_aspect('equal')

    plt.show()
    # return fig  # 需要保存