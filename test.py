from regression import get_x_data
import numpy as np

x_data = get_x_data()

gaussian_W = np.array([[ -6.932998  ,  22.3089   ,   -2.9852314 ],
 [ 30.308363 ,  -43.875942  ,  -4.2747426 ],
 [ 14.773739  ,  14.857093 ,  -17.948263  ],
 [ 10.313405  ,  40.40914  ,    0.04445635]])

sp_W = np.array([[-22.863413   ,  3.4204168  , -9.053357  ],
 [-20.854548 ,  -32.9337   ,   62.792767  ],
 [ 15.004759 ,    0.96945894 , -2.6903498 ],
 [-41.2841   ,  -11.8228445  , 28.635427  ]])

print('输入数据 : \n', x_data) #(9, 4)
print('高斯权值 : \n', gaussian_W) #(4, 3)
gaussian_xW = np.matmul(x_data, gaussian_W)
print('高斯近似结果 : \n', gaussian_xW)
print('\n\n\n')
print('输入数据 : \n', x_data) #(9, 4)
print('椒盐权值 : \n', sp_W) #(4, 3)
sp_xW = np.matmul(x_data, sp_W)
print('椒盐近似结果 : \n', sp_xW)