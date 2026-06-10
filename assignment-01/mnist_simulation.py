import tensorflow as tf
import pandas as pd

import tensorflow as tf
import pandas as pd


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0

def build_and_train_mlp(hidden_units, learning_rate=0.001):
    network = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_units, activation='tanh'), # Usamos activación Tanh en lugar de ReLU
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    

    network.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    
    # Entrenamiento con batch_size de 128
    network.fit(train_images, train_labels, epochs=4, batch_size=128, verbose=0)
    
    _, final_accuracy = network.evaluate(test_images, test_labels, verbose=0)
    return final_accuracy

# 3. Configuraciones
setups = [32, 64, 512]
summary_data = []

for units in setups:
    print(f"Running simulation with Hidden Units = {units}...")
    accuracy = build_and_train_mlp(units)
    summary_data.append({
        "Hidden Layer Architecture": f"[{units}x32]",
        "Optimization Algorithm": "RMSprop",
        "Accuracy Metric": f"{accuracy * 100:.2f}%"
    })

df = pd.DataFrame(summary_data)
print("\n=== VICTOR'S EXPERIMENTAL RESULTS ===")
print(df.to_string(index=False))

# Despliegue de la tabla de datos exclusiva de Víctor
df = pd.DataFrame(summary_data)
print("\n=== VICTOR'S EXPERIMENTAL RESULTS ===")
print(df.to_string(index=False))