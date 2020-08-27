import numpy as np
import matplotlib.pyplot as plt

loss = []
for i in range(400):
    loss.append(sum(np.loadtxt('./3dunet_model_save/loss_%d.txt' % i)))

y = np.array(loss)
y = y.flatten()
min = np.argmin(y)
# y = np.loadtxt('./3dunet_model_save/loss_99.txt')
print(min)
x = np.linspace(0,400-1,400)
# y = np.sin(x)
# print(x)

plt.plot(x, y, color='r', label=loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss_visualization')
# plt.legend()

plt.show()