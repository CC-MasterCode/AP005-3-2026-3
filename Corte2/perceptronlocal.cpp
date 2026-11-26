#include <iostream>
#include <fstream>
#include <cstdio>

using namespace std;

// definimos los cuatro patrones de entrenamiento con tres entradas no binarias y sus salidas esperadas
const int N = 4;
const double inputs[N][3] = {
    {2, 3, 4}, {3, 3, 4},
    {1, 2, 1}, {3, 1, 2}
};
const int targets[N] = {0, 1, 0, 1};

// inicializamos los tres pesos, el sesgo, la tasa de aprendizaje y el limite de epocas
double w1 = 0.0, w2 = 0.0, w3 = 0.0, b = 0.0;
const double eta = 0.2;
const int maxEpochs = 30;

// devolvemos 1 si la suma ponderada es mayor o igual a cero y 0 si es negativa
int activation(double z) { 
    return (z >= 0.0) ? 1 : 0; 
}

// multiplicamos cada entrada por su peso correspondiente y sumamos el sesgo
double weightedSum(double x1, double x2, double x3) {
    return w1 * x1 + w2 * x2 + w3 * x3 + b;
}

// calculamos la prediccion pasando la suma ponderada por la funcion escalon
int predict(double x1, double x2, double x3) {
    return activation(weightedSum(x1, x2, x3));
}

// recorremos los cuatro patrones sin alterar pesos para contar cuantos fallos existen al final de la epoca
int evaluate() {
    int errors = 0;
    for (int i = 0; i < N; i++) {
        double x1 = inputs[i][0];
        double x2 = inputs[i][1];
        double x3 = inputs[i][2];
        int t = targets[i];
        int y = predict(x1, x2, x3);
        
        if ((t - y) != 0) {
            errors++;
        }
    }
    return errors;
}

// ejecutamos el aprendizaje guardando en un archivo de texto los errores y los valores de z por epoca
void trainAndExport() {
    // abrimos el archivo donde se guardaran los datos numericos para graficar
    ofstream dataFile("datos_neurona.dat");
    dataFile << "# Epoca Errores_Eval Z_P1 Z_P2 Z_P3 Z_P4\n";

    // registramos el estado inicial antes de iniciar las actualizaciones
    int initialErrors = evaluate();
    dataFile << "0 " << initialErrors << " "
             << weightedSum(inputs[0][0], inputs[0][1], inputs[0][2]) << " "
             << weightedSum(inputs[1][0], inputs[1][1], inputs[1][2]) << " "
             << weightedSum(inputs[2][0], inputs[2][1], inputs[2][2]) << " "
             << weightedSum(inputs[3][0], inputs[3][1], inputs[3][2]) << "\n";

    // iteramos a traves de las epocas hasta que el error en evaluacion sea cero
    for (int epoch = 1; epoch <= maxEpochs; epoch++) {
        double z_epoch[N];

        // presentamos cada patron y actualizamos pesos y sesgo si hay error
        for (int i = 0; i < N; i++) {
            double x1 = inputs[i][0];
            double x2 = inputs[i][1];
            double x3 = inputs[i][2];
            int t = targets[i];

            double z = weightedSum(x1, x2, x3);
            z_epoch[i] = z; // guardamos la variable z de este patron antes de modificar los pesos
            int y = activation(z);
            int e = t - y;

            // aplicamos la regla delta para corregir los parametros en linea
            w1 += eta * e * x1;
            w2 += eta * e * x2;
            w3 += eta * e * x3;
            b += eta * e;
        }

        // evaluamos con los pesos fijos resultantes de la epoca
        int errors = evaluate();
        
        // escribimos la fila de resultados de la epoca en el archivo
        dataFile << epoch << " " << errors << " "
                 << z_epoch[0] << " " << z_epoch[1] << " " 
                 << z_epoch[2] << " " << z_epoch[3] << "\n";

        cout << "Epoca " << epoch << " | Errores en evaluacion: " << errors << endl;

        // detenemos el entrenamiento si se clasifican bien todos los patrones
        if (errors == 0) {
            cout << "Aprendizaje completado en la epoca " << epoch << endl;
            break;
        }
    }
    dataFile.close();
}

// abrimos una tuberia hacia gnuplot para dibujar las curvas directamente desde el archivo generado
void plotWithGnuplot() {
    FILE *gp = popen("gnuplot -persist", "w");
    if (gp) {import matplotlib.pyplot as plt

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
        fprintf(gp, "set title 'Aprendizaje de la Neurona (3 Entradas)'\n");
        fprintf(gp, "set xlabel 'Epocas'\n");
        fprintf(gp, "set ylabel 'Valor'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "plot 'datos_neurona.dat' using 1:2 with linespoints lw 2 title 'Errores por epoca', "
                    "'' using 1:3 with linespoints title 'Z Patron 1', "
                    "'' using 1:4 with linespoints title 'Z Patron 2', "
                    "'' using 1:5 with linespoints title 'Z Patron 3', "
                    "'' using 1:6 with linespoints title 'Z Patron 4'\n");
        pclose(gp);
    } else {
        cout << "Error al intentar abrir Gnuplot." << endl;
    }
}

// ejecutamos el entrenamiento y lanzamos la ventana grafica al finalizar
int main() {
    trainAndExport();
    plotWithGnuplot();
    return 0;
}