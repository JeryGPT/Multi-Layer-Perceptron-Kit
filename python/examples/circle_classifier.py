from pymlkit import Network, Activations, Losses
from random import random

RADIUS = 1
model = Network(2)
model.add_dense(16, Activations.ReLU)
model.add_dense(8, Activations.ReLU)
model.add_dense(1, Activations.Sigmoid)
model.set_loss(Losses.HalfMSE)
success_loaded = model.load("model.mlk")

if (not success_loaded):
    train_x = []
    train_y = []


    for i in range(0,10000):
        x = random() * 2 - 1
        y = random() * 2 - 1

        label = x * x + y*y < RADIUS * RADIUS
        train_x.append(x)
        train_x.append(y)
        train_y.append(float(label))

    model.fit(train_x, train_y, lr=0.1, epochs=100, batch=16, log_step=50)
SIZE = 20


for i in range(SIZE * SIZE):
    if i % SIZE == 0:
        print()

    grid_x = i % SIZE
    grid_y = i // SIZE

    x = grid_x / (SIZE - 1) * 2 - 1
    y = grid_y / (SIZE - 1) * 2 - 1

    prediction = model.forward([x, y])[0]

    print("██" if prediction > 0.5 else "  ", end="")

print()
if (not success_loaded):
    model.export()