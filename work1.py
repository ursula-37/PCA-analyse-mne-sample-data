#尝试PCA分析脑电数据
import numpy as np
import mne
from mne.datasets import sample
# sample的存放地址
data_path = sample.data_path()
# 该fif文件存放地址
fname = str(data_path) + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(fname)
data=raw.get_data()

def pca(data, n_components=None):
    x=data  #赋值
    #标准化处理
    mean=np.mean(x,axis=1,keepdims=True) #平均数
    std=np.std(x,axis=1,ddof=1,keepdims=True)#标准差，ddof=1代表采用样本标准差
    x1=(x-mean)/std  #第一步标准化x4
    x1=x1[~np.any(np.isinf(x1)|np.isnan(x1),axis=1)] #剔除坏道
    #计算标准化后数据的系数矩阵
    R=np.cov(x1)
    #特征值和特征向量
    x2,x3=np.linalg.eig(R) 
    y=x3@x1
    #选取主成分，第j个成分的贡献率
    b=x2/np.sum(x2)*100  #求贡献率
    for i in range(len(x2)):
            print(f'第{i+1}个成分的贡献率是{b[i]:.2f}%')
        
        
    b1=np.cumsum(b) #求贡献率的累加和
    print(f'前10成分累计贡献率为{b1[:10]}')
    return y
 
pca(data)



    
