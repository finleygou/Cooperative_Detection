import numpy as np

# angle_list = [6,3,4,5,2,0,7]
# print(np.sort(angle_list)[0])

# flag = [True, True, True, True,True, True]
#
# if all(flag) == False:
#     print('collide!!!!!!!')
#     r_l = -20
# else:
#     print('safe')

# data_ = ()
# data_ = data_ + (1., 2., 3., 4.)
# print(data_)
# data_ = data_ + (5,)
# print(data_)

def is_left(v1, v2):
    # 判断v2是否在v1的左边
    return np.cross(v1, v2) > 0

v1 = np.array([1, 0])
v2 = np.array([1, -2])

print(is_left(v1, v2))

a = np.random.randn(2)
# print(a)


# for i in range(4):
#         for j in range(i+1, 4):
#               print([i, j])
#             # for k in range(j+1, 4):
#             #     for l in range(k+1, 4):
#             #         print([i, j, k, l])

for i in range(100):
    print(np.random.randn(2))