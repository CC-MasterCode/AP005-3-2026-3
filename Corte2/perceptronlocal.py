import matplotlib.pyplot as plt

# definimos los cuatro patrones de entrenamiento con tres entradas no binarias y sus objetivos
inputs = [
    [2.0, 3.0, 4.0],
    [3.0, 3.0, 4.0],
    [1.0, 2.0, 1.0],
    [3.0, 1.0, 2.0]
]
targets = [0, 1, 0, 1]

# inicializamos los pesos, el sesgo, la tasa de aprendizaje y el limite de epocas
w1, w2, w3, b = 0.0, 0.0, 0.0, 0.0
eta = 0.2
max_epochs = 30

# devolvemos 1 si la suma ponderada es mayor o igual a cero y 0 en caso contrario
def activation(z):
    return 1 if z >= 0.0 else 0

# calculamos la combinacion lineal de entradas por pesos mas el sesgo
def weighted_sum(x1, x2, x3):
    return w1 * x1 + w2 * x2 + w3 * x3 + b

# pasamos los cuatro patrones con parametros fijos para contar los errores de evaluacion
def evaluate():
    errors = 0
    for i in range(len(inputs)):
        x1, x2, x3 = inputs[i]
        t = targets[i]
        y = activation(weighted_sum(x1, x2, x3))
        if (t - y) != 0:
            errors += 1
    return errors

# creamos listas para guardar el historial de epocas, errores y variables z
history_epochs = [0]
history_errors = [evaluate()]
history_z1 = [weighted_sum(*inputs[0])]
history_z2 = [weighted_sum(*inputs[1])]
history_z3 = [weighted_sum(*inputs[2])]
history_z4 = [weighted_sum(*inputs[3])]

# ejecutamos el ciclo principal de entrenamiento en linea
for epoch in range(1, max_epochs + 1):
    z_current_epoch = []

    # recorremos cada patron actualizando los parametros inmediatamente cuando hay error
    for i in range(len(inputs)):
        x1, x2, x3 = inputs[i]
        t = targets[i]

        z = weighted_sum(x1, x2, x3)
        z_current_epoch.append(z) # almacenamos la variable z antes de actualizar los pesos
        y = activation(z)
        e = t - y

        # ajustamos los pesos y el sesgo proporcionalmente al error y la tasa de aprendizaje
        w1 += eta * e * x1
        w2 += eta * e * x2
        w3 += eta * e * x3
        b += eta * e

    # evaluamos el rendimiento global tras terminar la ronda de actualizaciones
    errors = evaluate()
    print(f"Epoca {epoch} | Errores en evaluacion: {errors}")

    # guardamos los resultados de esta epoca en las listas de historial
    history_epochs.append(epoch)
    history_errors.append(errors)
    history_z1.append(z_current_epoch[0])
    history_z2.append(z_current_epoch[1])
    history_z3.append(z_current_epoch[2])
    history_z4.append(z_current_epoch[3])

    # salimos del bucle si la neurona ya clasifica correctamente todos los patrones
    if errors == 0:
        print(f"Aprendizaje completado en la epoca {epoch}")
        break

# configuramos y dibujamos las curvas de error y de la variable z usando matplotlib
plt.figure(figsize=(10, 6))
plt.plot(history_epochs, history_errors, marker='o', linewidth=2, color='red', label='Errores por epoca')
plt.plot(history_epochs, history_z1, marker='x', linestyle='--', label='Z Patron 1')
plt.plot(history_epochs, history_z2, marker='x', linestyle='--', label='Z Patron 2')
plt.plot(history_epochs, history_z3, marker='x', linestyle='--', label='Z Patron 3')
plt.plot(history_epochs, history_z4, marker='x', linestyle='--', label='Z Patron 4')

plt.title('Aprendizaje de la Neurona (3 Entradas)')
plt.xlabel('Epocas')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.show()