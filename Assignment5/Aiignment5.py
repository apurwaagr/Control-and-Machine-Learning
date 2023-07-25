import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Implement the neural network architecture
def create_model(hidden_units=10, activation='relu'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation=activation, input_shape=(2,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# Step 2: Prepare the data
x_train = np.array([[0.1, 0.1], [0.3, 0.4], [0.1, 0.5], [0.6, 0.9], [0.4, 0.2],
                    [0.6, 0.3], [0.5, 0.6], [0.9, 0.2], [0.4, 0.4], [0.7, 0.6]])
y_train = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

# Step 3: Define the optimization algorithms and parameters
optimizers = {
    'SGD': {'optimizer': tf.keras.optimizers.SGD, 'learning_rate': 0.05, 'momentum': 0.5},
    'Adagrad': {'optimizer': tf.keras.optimizers.Adagrad, 'learning_rate': 0.1, 'epsilon': 1e-8},
    'RMSprop': {'optimizer': tf.keras.optimizers.RMSprop, 'learning_rate': 0.005, 'epsilon': 1e-8},
    'Adam': {'optimizer': tf.keras.optimizers.Adam, 'learning_rate': 0.05,'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8}
}

# Step 4: Train the neural networks with different optimizers
def train_with_optimizer(optimizer_name, num_iterations=500):
    optimizer_params = optimizers[optimizer_name]
    optimizer = optimizer_params['optimizer'](learning_rate=optimizer_params['learning_rate'],
                                              **{k: v for k, v in optimizer_params.items() if k != 'optimizer' and k != 'learning_rate'})
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    objective_values = []

    for iteration in range(num_iterations):
        loss_value = model.train_on_batch(x_train, y_train)[0]
        objective_values.append(loss_value)

    return objective_values

# Step 5: Compare the results using a plot
plt.figure()

for optimizer_name in optimizers.keys():
    objective_values = train_with_optimizer(optimizer_name)
    plt.plot(objective_values, label=optimizer_name)

plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
plt.legend()
plt.show()
