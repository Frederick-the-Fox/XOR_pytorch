import numpy as np
import random
import torch
import matplotlib.pyplot as plt

#### 实验1：利用PyTorch的自动求导机制完成反向传播，并验证与DIY的结果是否一致
# Hint：所有的参数和运算都用torch中的数据形式和函数进行定义，数据也需要打包为tensor形式

#定义模型，并初始化参数（利用torch.tensor进行定义，并指明需要计算梯度：requires_grad=True）
def initialize_parameters(n_x, h_1, h_2, n_y):
    torch.manual_seed(2)

    # 模型参数用tensor表示，并随机初始化，或按某分布初始化
    #
    W1 = torch.randn(n_x, h_1, requires_grad=True, dtype=torch.float64)
    b1 = torch.randn(h_1, 1, requires_grad=True, dtype=torch.float64)
    W2 = torch.randn(h_1, h_2, requires_grad=True, dtype=torch.float64)
    b2 = torch.randn(h_2, 1, requires_grad=True, dtype=torch.float64)
    W3 = torch.randn(h_2,n_y, requires_grad=True, dtype=torch.float64)
    b3 = torch.randn(n_y, 1, requires_grad=True, dtype=torch.float64)
    # print(b1, b2, b3)
    # 将参数打包为dictionary形式
    # 例如： parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    return parameters



#前向传播
def forward_prop(X, parameters):
    # 从字典dictionary里读取出所有的参数
    # 例如 W1=parameters["W1"]


    # 利用torch中的tensor运算，依次计算每层的z和a
    # 例如：
    # z1 = torch.add(torch.matmul(W, X), b1)
    # a1 = 1/(1+torch.exp(-z1))

    z1 = torch.add(torch.matmul(parameters['W1'].T, X), parameters['b1'])
    a1 = 1/ ( 1 + torch.exp(-z1))
    z2 = torch.add(torch.matmul(parameters['W2'].T, a1), parameters['b2'])
    a2 = 1 / (1 + torch.exp(-z2))
    z3 = torch.add(torch.matmul(parameters['W3'].T, a2), parameters['b3'])
    a3 = 1 / (1 + torch.exp(-z3))

    y_pred = a3      # 输出

    return y_pred


#损失函数计算
def calculate_cost(y_pred,Y):

    # 定义损失函数，均方根或交叉熵
    cost = (1/2 * torch.sum((y_pred[0,] - Y)**2)) / len(Y)

    return cost

def calculate_cost_L2(y_pred, Y, parameter):

    cost = (1 / 2 * torch.sum((y_pred[0,] - Y) ** 2)) / len(Y) \
           + 0.001 * (torch.norm(parameter['W1'])
                       + torch.norm(parameter['W2'])
                       + torch.norm(parameter['W3'])
                       + torch.norm(parameter['b1'])
                       + torch.norm(parameter['b2'])
                       + torch.norm(parameter['b3']))

    return cost

# # 利用训练完的模型进行预测
# def predict(X, parameters):
#
#
#     return y_predict


# 主程序
if __name__ == '__main__':
    # np.random.seed(2)

    # 准备训练数据和标签
    train_num = 100
    test_num = 100

    X_train_0 = np.random.rand(2, train_num)
    Y_train_0 = np.zeros(train_num).T

    X_train_1 = -np.random.rand(2, train_num)
    Y_train_1 = np.zeros(train_num).T

    X_train_2 = np.random.rand(2, train_num)
    X_train_2[0,:] = -X_train_2[0,:]
    Y_train_2 = np.ones(train_num).T

    X_train_3 = np.random.rand(2, train_num)
    X_train_3[1,:] = -X_train_3[1,:]
    Y_train_3 = np.ones(train_num).T

    X_train = np.concatenate((X_train_0, X_train_1, X_train_2, X_train_3), axis=1)
    Y_train = np.concatenate((Y_train_0, Y_train_1, Y_train_2, Y_train_3), axis=0)

    # 画出训练数据的图

    plt.figure(1)
    plt.scatter(X_train[0,], X_train[1,], c=list(Y_train), cmap='jet')
    plt.show()

    # 包装为tensor

    X_train, Y_train = torch.tensor(X_train), torch.tensor(Y_train)

    # 超参数设置

    # num_of_iters = 10000     # 迭代次数
    num_of_epochs = 100      # 迭代轮数
    learning_rate = 1    # 学习率
    batch_size = 1         # batch size

    # 模型定义与初始化 # n_x, n_y: 输入/输出的神经元个数，中间是自己定义的MLP神经元个数

    init_params = initialize_parameters(n_x=2, h_1=5, h_2=5, n_y=1)
    parameters = init_params

    # # 打乱数据顺序
    #
    # idx = list(range(train_num * 4))
    # random.shuffle(idx)
    # X_train = X_train[:,idx]
    # Y_train = Y_train[idx]

    # 训练模型

    plt.figure(1)
    cost_iter = [] # 保存损失函数的变化

    # for i in range(0, num_of_iters): # 每次迭代
    for epoch in range(num_of_epochs):
        # 装载数据-------Dataloader
        # 打乱数据顺序
        idx = list(range(train_num * 4))
        random.shuffle(idx)
        # print(idx)
        X_train = X_train[:, idx]
        Y_train = Y_train[idx]

        for i in range(int(train_num * 4 / batch_size)):
            # 取一个mini-batch的数据
            data_start = i * batch_size
            X_minibatch = X_train[:,data_start:data_start + batch_size]
            Y_minibatch = Y_train[data_start:data_start + batch_size]

            # 正向传播
            y_pred = forward_prop(X_minibatch, parameters)

            # 计算损失函数
            # cost = calculate_cost(y_pred, Y_minibatch)
            cost = calculate_cost_L2(y_pred, Y_minibatch, parameters)
            # 梯度下降并更新参数
            # 自动求导

            cost.backward()

            # 查看自动求导之后各参数的梯度，例如：print("dW1:", W1.grad.numpy())，检验是否与DIY的结果一致
            # print("dW1:", parameters['W1'].grad.numpy())

            with torch.no_grad():
                # 参数更新，例如 W1 -= learning_rate*W1.grad
                parameters['W1'] -= learning_rate * parameters['W1'].grad
                parameters['b1'] -= learning_rate * parameters['b1'].grad
                parameters['W2'] -= learning_rate * parameters['W2'].grad
                parameters['b2'] -= learning_rate * parameters['b2'].grad
                parameters['W3'] -= learning_rate * parameters['W3'].grad
                parameters['b3'] -= learning_rate * parameters['b3'].grad
            # 所有参数的梯度归零，例如W1.grad.zero_()（思考为什么要做这个）

            parameters['W1'].grad.zero_()
            parameters['b1'].grad.zero_()
            parameters['W2'].grad.zero_()
            parameters['b2'].grad.zero_()
            parameters['W3'].grad.zero_()
            parameters['b3'].grad.zero_()

            # 可设置迭代停止条件

            # 观察损失函数下降情况

        # print('cost after epoch #{:d}:{:f}'.format(epoch, cost))
        cost_iter.append(cost.detach().numpy())

    plt.figure(2)
    plt.plot(cost_iter)
    plt.show()
    #
    # 得到最终的参数值，例如：W1_star = W1.detach()


    # 测试数据
    # np.random.seed(2)

    X_test_0 = np.random.rand(2, test_num)
    Y_test_0 = np.zeros(test_num).T

    X_test_1 = -np.random.rand(2, test_num)
    Y_test_1 = np.zeros(test_num).T

    X_test_2 = np.random.rand(2, test_num)
    X_test_2[0, :] = -X_test_2[0, :]
    Y_test_2 = np.ones(test_num).T

    X_test_3 = np.random.rand(2, test_num)
    X_test_3[1, :] = -X_test_3[1, :]
    Y_test_3 = np.ones(test_num).T

    X_test = np.concatenate((X_test_0, X_test_1, X_test_2, X_test_3), axis=1)
    Y_test = np.concatenate((Y_test_0, Y_test_1, Y_test_2, Y_test_3), axis=0)

    X_test, Y_test = torch.tensor(X_test), torch.tensor(Y_test)

    plt.figure(3)
    plt.scatter(X_test[0,], X_test[1,], c=list(Y_test), cmap='jet')
    plt.show()

    with torch.no_grad():
        Y_train_pred = forward_prop(X_train, parameters)
        Y_test_pred = forward_prop(X_test, parameters)

    Y_train_pred[Y_train_pred >= 0.5] = 1
    Y_train_pred[Y_train_pred < 0.5] = 0
    Y_test_pred[Y_test_pred >= 0.5] = 1
    Y_test_pred[Y_test_pred < 0.5] = 0


    print('train acc', torch.sum(Y_train_pred[0,] == Y_train) / (train_num * 4))
    print('test acc', torch.sum(Y_test_pred[0,] == Y_test) / (test_num * 4))


