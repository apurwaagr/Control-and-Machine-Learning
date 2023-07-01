import numpy as np
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')
plt.figure(figsize=(8,8))
def Sigmoid(t):
    return 1/(1+np.exp(-t))
def HyperbolicTan(t):
    return np.tanh(t)
def ReLu(t):
    list = []
    for i in t:
        if i>=0:
            list.append(i)
        else:
            list.append(0)
    return list

def LeakyRectifiedLinearUnit(t):
    lst=[]
    for i in t:
        if i>=0:
            lst.append(i)
        else:
            lst.append(0.01*i)
    return lst

def softmax(t):
    return np.exp(t) / np.sum(np.exp(t))

t = np.linspace(-5,5)
plt.plot(t, Sigmoid(t), color="#307EC7", label ="sigmoid")
plt.plot(t, HyperbolicTan(t), color="#9621E2", label = "derivative")
plt.plot(t, ReLu(t), color="#808080", label = "ReLu")
plt.plot(t, LeakyRectifiedLinearUnit(t), color="#FFA500", label = "LeakyRectifiedLinearUnit",linestyle="dashed")
plt.plot(t, softmax(t), color="#00FF00", label = "softmax")
plt.legend(loc="upper right", frameon=True)
plt.title('Activation Functions')
plt.show()
