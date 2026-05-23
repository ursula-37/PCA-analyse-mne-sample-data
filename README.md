# PCA-analyse-mne-sample-data
课程上的一个作业，刚好把数学建模课和python可视化课程结合一下
PCA分析脑电数据。
数据背景：
这些数据是通过Neuromag获得的 MGH/HMS/MIT阿西努拉·A·马蒂诺斯生物医学中心的矢量视图系统 影像学。同时采集了来自60通道电极帽的脑电图数据， MEG。原始MRI数据集是用西门子1.5 T相机采集的 使用MPRAGE序列的Sonata扫描仪。在本次实验中，向受试者展示了棋盘格图案 进入左右视野，穿插着音调到 左耳或右耳。刺激之间的间隔为750毫秒。偶尔 在视野中央展示了一个笑脸。 受试者被要求用右手食指在脸部出现后尽快摁下按键。


前10成分累计贡献率为[19.98489093 36.11354557 40.91608815 45.24186208 48.57653186 51.35645753
 53.34937681 55.2694206  57.00336572 58.49640083]

 
点评：
通过PCA可以看出数据是否可用。本次数据下降速度快，数据高度冗余，通道间数据相关。PC1和PC2为噪声。两个数据可能为心跳伪迹和眨眼伪迹，之后分析可以予以排除。可以对他进行降维，保留约50个成分就可以。
而且，发现numpy功能超级强大，对于数学上的一些式子已经封装好了，直接用就可以。比如求相关系数矩阵可以直接使用np.cov()。



代码：
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
 
pca(data)
