import matplotlib.pyplot as plt

from src.numberGenerator.chaos.dissipative import Dissipative
from src.numberGenerator.chaos.lozi import Lozi
from src.numberGenerator.chaos.tinkerbell import Tinkerbell

cprng = Tinkerbell()

plt.grid(1)
plt.xlabel('X')
plt.ylabel('Y')
plt.ion()

for _ in range(10000):
    result = cprng.getNext()
    plt.scatter(result[0], result[1], color='black', s=0.1, label="test1")

plt.show(5)

plt.close()

cprng = Lozi()

plt.grid(1)
plt.xlabel('X')
plt.ylabel('Y')
plt.ion()

for _ in range(10000):
    result = cprng.getNext()
    plt.scatter(result[0], result[1], color='black', s=0.1, label="test1")

plt.show(5)

cprng = Dissipative()

plt.grid(1)
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim([0, 6])
plt.xlim([0, 6])
plt.ion()

for _ in range(10000):
    result = cprng.getNext()
    plt.scatter(result[0], result[1], color='black', s=0.1, label="test1")

plt.show(5)
