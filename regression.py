import tensorflow as tf
import mean_std
import numpy as np
from sklearn.preprocessing import LabelBinarizer #标签二值化
from sklearn.model_selection import train_test_split #分割数据集


print('数据处理...')

#shape是(9, )
images_means = mean_std.get_images_mean()
images_stds = mean_std.get_images_std()
print('平均值 : ', images_means)
print('标准差 : ', images_stds)
max_points = mean_std.max_point()
min_points = mean_std.min_point()
print('最大点 : ', max_points)
print('最小点 : ', min_points)

#(9, 4)数据集,9行依次是第一张图片到第九张图片
x_data = np.vstack((images_means, images_stds, max_points, min_points)).T
#标准化
x_data -= x_data.min()
x_data /= x_data.max()
print('x数据 : ', x_data)

def get_x_data():
    return x_data

#高斯噪声九张图片的标签
#单层字典为0,深层字典为1,自动编码器为2
'''
图片1: 		高斯:单>深>自 	椒盐:深>自>单
图片2:		高斯:单>自>深    	椒盐:单>自>深
图片3:		高斯:深>自>单	    椒盐:自>深>单
图片4:		高斯:单>深>自	    椒盐:单>深>自
图片5:		高斯:深>单>自	    椒盐:深>自>单
图片6:		高斯:单>自>深	    椒盐:自>深>单
图片7:		高斯:单>深>自	    椒盐:深>自>单
图片8:		高斯:单>自>深	    椒盐:自>深>单
图片9:		高斯:单>深>自	    椒盐:深>自>单
'''
gaussian_label = ['0', '0', '1', '0', '1', '0', '0', '0', '0']
print('高斯噪声标签 : ', gaussian_label)
#椒盐噪声的标签
sp_label = ['1', '0', '2', '0', '1', '2', '1', '2', '1']
print('椒盐噪声标签 : ', sp_label)

#二值化标签
gaussian_label.append('2')
gaussian_one_hot_label = LabelBinarizer().fit_transform(gaussian_label)
gaussian_label.pop()
#shape: (9, 3)
gaussian_one_hot_label = gaussian_one_hot_label[:-1, :]
#shape: (9, 3)
sp_one_hot_label = LabelBinarizer().fit_transform(sp_label)


print('开始训练...')

x = tf.placeholder(tf.float32, shape=(None, 4)) #4个因素影响结果
y = tf.placeholder(tf.float32, shape=(None, 3)) #3分类问题
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(dtype=tf.float32, initial_value=1e-3) #学习率的初值是0.001

#3层BP网络模型(4, 200, 3)
hidden_num = 200 #隐层神经元个数
#第一层
w1 = tf.Variable(tf.truncated_normal((4, hidden_num)), dtype=tf.float32)
b1 = tf.Variable(tf.zeros((hidden_num)) + 0.1, dtype=tf.float32)
L1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)
#第二层
w2 = tf.Variable(tf.truncated_normal((hidden_num, 3)), dtype=tf.float32)
b2 = tf.Variable(tf.zeros((3)) + 0.1, dtype=tf.float32)
prediction = tf.matmul(L1_drop, w2) + b2
prediction_one_hot = tf.nn.softmax(prediction)

#代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits=prediction))
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

#准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# #50个隐层神经元, 1500次step, 1.0dropout系数, 学习率衰减系数0.9995
# print('开始训练高斯标签...')
# #高斯噪声
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(1500):
#         #每次训练将学习率调低
#         sess.run(tf.assign(lr, 1e-3 * (0.9995 ** step)))
#         learning_rate, _ = sess.run([lr, train_step],
#                                     feed_dict={x: x_data, y: gaussian_one_hot_label, keep_prob: 1.0})
#         # print('Step : ', step, ' Learning Rate : ', learning_rate)
#         if step % 10 == 0: #每训练100次
#             Loss, acc_train = sess.run([loss, accuracy],
#                                  feed_dict={x: x_data, y: gaussian_one_hot_label, keep_prob: 1.0})
#             print('Step : ', step, ' Accuracy : ', acc_train, ' Loss : ', Loss)
#
#     #获取权值矩阵(4, 200)
#     W1 = sess.run(w1, feed_dict={x: x_data, y: gaussian_one_hot_label, keep_prob: 1.0})
#     #获取权值矩阵(200, 3)
#     W2 = sess.run(w2, feed_dict={x: x_data, y: gaussian_one_hot_label, keep_prob: 1.0})
#     W = np.matmul(W1, W2)
#     print('权值矩阵 : ', W)
# print('完成训练高斯标签...')


# # 200个隐层神经元, 1500次step, 1.0dropout系数, 学习率衰减系数0.9999
# print('开始训练椒盐标签...')
# #椒盐噪声
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(1500):
#         #每次训练将学习率调低
#         sess.run(tf.assign(lr, 1e-3 * (0.9999 ** step)))
#         learning_rate, _ = sess.run([lr, train_step],
#                                     feed_dict={x: x_data, y: sp_one_hot_label, keep_prob: 1.0})
#         # print('Step : ', step, ' Learning Rate : ', learning_rate)
#         if step % 10 == 0: #每训练100次
#             Loss, acc_train = sess.run([loss, accuracy],
#                                  feed_dict={x: x_data, y: sp_one_hot_label, keep_prob: 1.0})
#             print('Step : ', step, ' Accuracy : ', acc_train, ' Loss : ', Loss)
#
#     #获取权值矩阵(4, 200)
#     W1 = sess.run(w1, feed_dict={x: x_data, y: sp_one_hot_label, keep_prob: 1.0})
#     #获取权值矩阵(200, 3)
#     W2 = sess.run(w2, feed_dict={x: x_data, y: sp_one_hot_label, keep_prob: 1.0})
#     W = np.matmul(W1, W2)
#     print('权值矩阵 : ', W)
# print('完成训练椒盐标签...')

print('Finish')