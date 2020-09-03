import numpy as np
import matplotlib.pyplot as plt

loss = []
for i in range(200):
    loss.append(sum(np.loadtxt('./3dunet_model_save/loss_%d.txt' % i)))

y = np.array(loss)
y = y.flatten()
# y = np.loadtxt('./3dunet_model_save/loss_99.txt')
# print(y)
x = np.linspace(0,200-1,200)
# y = np.sin(x)
# print(x)

plt.plot(x, y, color='r', label=loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss_visualization')
# plt.legend()

plt.show()