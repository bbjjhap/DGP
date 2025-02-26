import time

import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, PolynomialKernel, RQKernel, SpectralMixtureKernel, \
    ProductStructureKernel, GridInterpolationKernel, CosineKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

import urllib.request
import os
from scipy.io import loadmat
from math import floor
import pandas as pd
import numpy as np
import tensorflow as tf


# this is for running the notebook in our testing framework
from sklearn.metrics import median_absolute_error, explained_variance_score, mean_squared_error, mean_absolute_error, \
    r2_score

smoke_test = ('CI' in os.environ)





# important file












# if not smoke_test and not os.path.isfile('../elevators.mat'):
#     print('Downloading \'elevators\' UCI dataset...')
#     urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')
#
#
# if smoke_test:  # this is for running the notebook in our testing framework
#     X, y = torch.randn(1000, 3), torch.randn(1000)
# else:
#     data = torch.Tensor(loadmat('D:\CjlNoFile\组会文件\深度高斯过程\\3.4集成异核高斯模型\\3.4集成异核高斯模型\data\elevators.mat')['data'])
#     X = data[:, :-1]
#     X = X - X.min(0)[0]
#     X = 2 * (X / X.max(0)[0]) - 1
#     y = data[:, -1]
#
#
# train_n = int(floor(0.8 * len(X)))
# train_x = X[:train_n, :].contiguous()
# train_y = y[:train_n].contiguous()
# print(train_x.dtype)
# print('train_x.shape :',train_x.shape)
# print('train_y.shape :',train_y.shape)
# test_x = X[train_n:, :].contiguous()
# test_y = y[train_n:].contiguous()


#dataframe = pd.read_csv('D:\CJL\DGP\\3.4集成异核高斯模型\\3.4集成异核高斯模型\data\jiediansunhao-cjl.CSV',header=None)
dataframe = pd.read_csv('E:\CjlFile\组会文件\深度高斯过程\DGP\\3.4集成异核高斯模型\\3.4集成异核高斯模型\data\dianji_cjl.CSV',header=None)


dataset = dataframe.values
row, column = dataset.shape
print(row, column)
dataset = dataset[:,0:column]
# for battery_prediction
# dataset = dataset[:,1:column]
df2 = dataframe.copy()
# print("\n原始数据:\n",df2)
df2 = df2.values


X = df2[:,0:column-1]
Y = df2[:,column-1:column]
from sklearn.preprocessing import scale, MinMaxScaler,StandardScaler
# X = scale(X)
# Y = scale(Y)
mm = MinMaxScaler()
X = mm.fit_transform(X)
Y = mm.fit_transform(Y)
# ss = StandardScaler()
# X = ss.fit_transform(X)
# Y = ss.fit_transform(Y)
X = X.astype(np.float32)
Y = Y.astype(np.float32)
print(X.dtype)
X = torch.tensor(X)
Y = torch.tensor(Y)

print(X.dtype)
# train_n = int(floor(0.8 * len(X)))
# train_x = X[:train_n, :].contiguous()
# train_y = Y[:train_n].contiguous()
# #train_y = torch.squeeze(train_y,1)
# print(type(train_x))
# print('train_x.shape :',train_x.shape)
# print('train_y.shape :',train_y.shape)
# test_x = X[train_n:, :].contiguous()
# test_y = Y[train_n:].contiguous()
#test_y = torch.squeeze(test_y,1)

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = Y[:train_n].contiguous()
train_y = torch.squeeze(train_y,1)
print(type(train_x))
print('train_x.shape :',train_x.shape)
print('train_y.shape :',train_y.shape)
test_x = X[train_n:, :].contiguous()
test_y = Y[train_n:].contiguous()
test_y = torch.squeeze(test_y,1)


if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
#n this notebook, we provide a GPyTorch implementation of deep Gaussian processes,
# where training and inference is performed using the method of Salimbeni et al., 2017
# (https://arxiv.org/abs/1705.08933) adapted to CG-based inference.
class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            #inducing_points = train_x[torch.randperm(train_x.size(0))[:2000]]
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        # self.covar_module = ScaleKernel(
        #     MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
        #     batch_shape=batch_shape, ard_num_dims=None
        # )
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        #     grid_size=100, num_dims=2,
        # )
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
        #self.covar_module = RBFKernel()
        #self.covar_module = MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):

        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

num_output_dims = 2 if smoke_test else 10


class DeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )
        hidden_layer2 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer1.output_dims,
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer3 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer2.output_dims +train_x_shape[-1] ,
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer4 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer3.output_dims,
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer5 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer4.output_dims+train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer6 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer5.output_dims,
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer7 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer6.output_dims+train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer8 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer7.output_dims,
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer9 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer8.output_dims+train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer10 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer9.output_dims,
            output_dims=num_output_dims,
            mean_type='constant',
        )
        hidden_layer11 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer10.output_dims+train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='constant',
        )


        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer11.output_dims,
            output_dims= None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.hidden_layer3 = hidden_layer3
        self.hidden_layer4 = hidden_layer4
        self.hidden_layer5 = hidden_layer5
        self.hidden_layer6 = hidden_layer6
        self.hidden_layer7 = hidden_layer7
        self.hidden_layer8 = hidden_layer8
        self.hidden_layer9 = hidden_layer9
        self.hidden_layer10 = hidden_layer10
        self.hidden_layer11= hidden_layer11
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer1(inputs)
        hidden_rep2 = self.hidden_layer2(hidden_rep1)
        hidden_rep3 = self.hidden_layer3(hidden_rep2,inputs)
        hidden_rep4 = self.hidden_layer4(hidden_rep3)
        hidden_rep5 = self.hidden_layer5(hidden_rep4,inputs)
        hidden_rep6 = self.hidden_layer6(hidden_rep5)
        hidden_rep7 = self.hidden_layer7(hidden_rep6,inputs)
        hidden_rep8 = self.hidden_layer8(hidden_rep7)
        hidden_rep9 = self.hidden_layer9(hidden_rep8,inputs)
        hidden_rep10 = self.hidden_layer10(hidden_rep9)
        hidden_rep11= self.hidden_layer11(hidden_rep10,inputs)
        output = self.last_layer(hidden_rep11)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:

                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


model = DeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()
# this is for running the notebook in our testing framework
num_epochs = 1 if smoke_test else 20
num_samples = 3 if smoke_test else 10


optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
train_start = time.time()
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibar", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            #print(x_batch)
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            minibatch_iter.set_postfix(loss=loss.item())
train_end = time.time() - train_start
print(f"Time to train Delta: {train_end:.2f}s")
import gpytorch
import math


test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=128)

model.eval()
start_time = time.time()
predictive_means, predictive_variances, test_lls = model.predict(test_loader)

pred_time = time.time() - start_time
#print(f"Time to compute dgp mean + covariances===before: {pred_time:.2f}s")
# for i in epochs_iter:
#     # Within each iteration, we will go over each minibatch of data
#     minibatch_iter = tqdm.tqdm(train_loader, desc="Minibar", leave=False)
#     for x_batch, y_batch in minibatch_iter:
#         with gpytorch.settings.num_likelihood_samples(num_samples):
#             optimizer.zero_grad()
#             #print(x_batch)
#             output = model(x_batch)
#             loss = -mll(output, y_batch)
#             loss.backward()
#             optimizer.step()
#
#             minibatch_iter.set_postfix(loss=loss.item())
model.eval()


print(f"Time to compute dgp mean + covariances===before: {pred_time:.2f}s")

rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")



test_y = torch.unsqueeze(test_y,1)


print('test_y.shape :',test_y.shape)
test_y = test_y.cpu()
predictive_means = predictive_means.cpu()


# print(predictive_means.mean().cpu())VariationalELBO--Scalable Variational Gaussian Process Classification
# print(mean.mean())
# print(test_y.mean())
# R2 = r2_score(ty,pmean, multioutput='raw_values')  # 拟合优度
# R22 = 1 - tf.sqrt(1 - R2)
from dgp_model.metric_caculate import R2_score
R2 = R2_score(test_y,predictive_means.mean(0))
R22 = 1 - tf.sqrt(1 - R2)
time12 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("当前模型训练结束，时间为： "+time12)
print(f"Time to train : {train_end:.2f}s")
print("R2_gpytorch_test : " , R2)
print("R22 : " , R22)
Mse = mean_squared_error(test_y, predictive_means.mean(0))  # 均方差
Mae = mean_absolute_error(test_y, predictive_means.mean(0),
                              sample_weight=None,
                              multioutput='uniform_average')  # 平均绝对误差
Variance = explained_variance_score(test_y, predictive_means.mean(0),
                                        sample_weight=None,
                                        multioutput='uniform_average')  # 可释方差得分
Meae = median_absolute_error(test_y, predictive_means.mean(0))  # 中值绝对误差
# R2 = r2_score(y_test, y_pred, multioutput='raw_values')  # 拟合优度
# R22 = 1 - tf.sqrt(1 - R2)
# Mse = mean_squared_error(y_test, y_pred)  # 均方差
# Mae = mean_absolute_error(y_test, y_pred,
#                           sample_weight=None,
#                           multioutput='uniform_average')  # 平均绝对误差
# Variance = explained_variance_score(y_test, y_pred,
#                                     sample_weight=None,
#                                     multioutput='uniform_average')  # 可释方差得分
# Meae = median_absolute_error(y_test, y_pred)  # 中值绝对误差

# print("R2_gpytorch_test : " , R2)
# print("R22 : " , R22)
print("Mse :",  Mse)
print("Rmse :",  np.sqrt(Mse))
print("Mae :",  Mae)
#print("Variance :",  Variance)
print("Meae :",  Meae)
print('=====================================')
# R2 = r2_score(ty, mm, multioutput='raw_values')  # 拟合优度
# R22 = 1 - tf.sqrt(1 - R2)

# R2 = r2_score(y_test, y_pred, multioutput='raw_values')  # 拟合优度
# R22 = 1 - tf.sqrt(1 - R2)
# Mse = mean_squared_error(y_test, y_pred)  # 均方差
# Mae = mean_absolute_error(y_test, y_pred,
#                           sample_weight=None,
#                           multioutput='uniform_average')  # 平均绝对误差
# Variance = explained_variance_score(y_test, y_pred,
#                                     sample_weight=None,
#                                     multioutput='uniform_average')  # 可释方差得分
# Meae = median_absolute_error(y_test, y_pred)  # 中值绝对误差

# print("R2_gpytorch_test : " , R2)
# print("R22 : " , R22)
print("Mse :",  Mse)
print("Rmse :",  np.sqrt(Mse))
print("Mae :",  Mae)
#print("Variance :",  Variance)
print("Meae :",  Meae)
print('=====================================')
path = 'E:\CjlFile\组会文件\深度高斯过程\DGP\\3.4集成异核高斯模型\\3.4集成异核高斯模型\code\\test\dgp_model\\result\\dianji_result.txt'
txt_file = open(path,'a')
time12 = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
file_name = 'DGP_ScaleRBF_12layer_Cholesky_short_Epoch20'
txt_file.write(file_name+' '+time12+'\n')
txt_file.write(f"Time to train: {train_end:.2f}s\n")
txt_file.write('Mse :'+  str(Mse)+'\n')
txt_file.write('Rmse :'+ str( np.sqrt(Mse))+'\n')
txt_file.write('Mae :'+  str(Mae)+'\n')
txt_file.write('Meae :'+  str(Meae)+'\n')
txt_file.write('        \n')
txt_file.close()