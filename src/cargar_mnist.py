"""
Carga y visualización del dataset MNIST con scikit-learn.

Este script:
    1. Descarga (la primera vez) MNIST mediante `fetch_openml`.
    2. Separa los datos en entrenamiento y prueba.
    3. Imprime las formas (dimensiones) de los arrays resultantes.
    4. Muestra 15 imágenes de dígitos en una grilla con matplotlib.

Ejecutar con:
    python src/cargar_mnist.py

Nota: la primera ejecución necesita conexión a Internet para descargar
MNIST (~11 MB). Las siguientes ejecuciones usan la copia en caché.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def cargar_mnist():
    """Descarga MNIST desde OpenML y devuelve las imágenes y sus etiquetas.

    - X: matriz de imágenes. Cada fila es una imagen aplanada de 28x28 = 784
         píxeles, con valores entre 0 (negro) y 255 (blanco).
    - y: vector con la etiqueta (dígito del 0 al 9) de cada imagen.

    `as_frame=False` hace que scikit-learn devuelva arrays de NumPy en lugar
    de un DataFrame de pandas. pandas se usa internamente para leer los datos
    de OpenML, por eso está incluido en requirements.txt.
    """
    print("Cargando MNIST (la primera vez puede tardar un poco)...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
    )

    X = mnist.data                # Forma esperada: (70000, 784)
    y = mnist.target.astype(int)  # Etiquetas como enteros (0-9)
    return X, y


def separar_datos(X, y):
    """Separa el dataset en conjunto de entrenamiento y de prueba.

    Se reservan 10 000 ejemplos para prueba (el tamaño estándar de MNIST)
    y el resto para entrenamiento. stratify=y mantiene la misma proporción
    de cada dígito en ambos conjuntos.

    random_state fija la semilla para que la separación sea siempre la misma.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=10_000,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def mostrar_formas(X_train, X_test, y_train, y_test):
    """Imprime las dimensiones de los arrays para verificar la separación."""
    print("\nFormas de los arrays:")
    print(f"  X_train: {X_train.shape}  -> imágenes de entrenamiento")
    print(f"  y_train: {y_train.shape}  -> etiquetas de entrenamiento")
    print(f"  X_test:  {X_test.shape}   -> imágenes de prueba")
    print(f"  y_test:  {y_test.shape}   -> etiquetas de prueba")

    # Datos extra útiles para entender el dataset.
    print(f"\nCantidad de clases: {len(np.unique(y_train))}")
    print(f"Rango de píxeles: {X_train.min()} a {X_train.max()}")


def mostrar_digitos(X, y, cantidad=15):
    """Muestra `cantidad` imágenes de dígitos en una grilla de 3 filas x 5 columnas.

    X contiene imágenes aplanadas (784 valores). Para dibujarlas hay que
    volver a darles su forma original de 28x28 con reshape(28, 28).
    """
    filas, columnas = 3, 5
    fig, ejes = plt.subplots(filas, columnas, figsize=(10, 6))

    for i, eje in enumerate(ejes.flat):
        if i >= cantidad:
            # Si sobran ejes en la grilla, los ocultamos.
            eje.axis("off")
            continue

        # Reconstruimos la imagen 28x28 y la dibujamos en escala de grises.
        imagen = X[i].reshape(28, 28)
        eje.imshow(imagen, cmap="gray")

        # Mostramos la etiqueta real encima de cada imagen.
        eje.set_title(f"Etiqueta: {y[i]}")
        eje.axis("off")  # Quitamos los ejes (coordenadas) para una vista limpia

    fig.suptitle("15 ejemplos del dataset MNIST", fontsize=14)
    plt.tight_layout()
    plt.show()  # Abre la ventana con la grilla de imágenes


def main():
    """Función principal: ejecuta todos los pasos en orden."""
    X, y = cargar_mnist()
    X_train, X_test, y_train, y_test = separar_datos(X, y)
    mostrar_formas(X_train, X_test, y_train, y_test)
    mostrar_digitos(X_train, y_train, cantidad=15)


if __name__ == "__main__":
    main()
