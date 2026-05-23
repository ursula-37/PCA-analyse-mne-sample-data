import matplotlib.pyplot as plt
from work1 import data, pca

y = pca(data)
brain=y[0:5, :]
plt.figure()
for i in range(5):
    plt.plot(brain[i, :5000] + i*10, label=f'PC{i+1}')  # 偏移避免重叠
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('PC3-5:is it brain signal?')
plt.legend()
plt.show()