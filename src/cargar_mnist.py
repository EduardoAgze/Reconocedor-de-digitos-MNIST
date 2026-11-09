"""
Carga y visualización del dataset MNIST con scikit-learn.

Este script:
    1. Descarga (la primera vez) MNIST mediante `fetch_openml`.
    2. Separa los datos en entrenamiento y prueba.
    3. Imprime las formas (dimensiones) de los arrays resultantes.
    4. Normaliza los píxeles al rango [0, 1] dividiendo entre 255.0.
    5. Muestra el mínimo y máximo de los datos antes y después.
    6. Compara una misma imagen antes y después de normalizar.
    7. Muestra 15 imágenes de dígitos en una grilla con matplotlib.

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

    # Dato extra útil para entender el dataset.
    print(f"\nCantidad de clases: {len(np.unique(y_train))}")


def mostrar_rango(X, nombre):
    """Imprime el valor mínimo y máximo de un array de imágenes."""
    print(f"  {nombre}: mínimo = {X.min():.1f}, máximo = {X.max():.1f}")


def normalizar_datos(X_train, X_test):
    """Normaliza los píxeles al rango [0, 1] dividiendo entre 255.0.

    ¿Por qué normalizar?
    - Los píxeles originales van de 0 a 255. Al dividirlos entre 255.0 todas
      las entradas quedan en una escala pequeña y comparable.
    - Muchos modelos, especialmente las redes neuronales, aprenden más rápido
      y de forma más estable con entradas en este rango. Así los valores
      grandes no dominan los cálculos ni desestabilizan los gradientes.
    - Se aplica el mismo factor a entrenamiento y prueba para que ambos
      conjuntos queden en la misma escala. El 255.0 es el máximo posible de un
      píxel, no se calcula con los datos; por eso no hay fuga de información.
    - Normalizar no cambia el contenido de la imagen, solo su escala numérica.

    Antes de dividir se convierte a float32: los datos cargados con pandas
    pueden ser enteros, y necesitamos valores decimales. float32 ocupa la
    mitad de memoria que float64.
    """
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255.0
    X_test /= 255.0
    return X_train, X_test


def comparar_normalizacion(imagen_antes, imagen_despues, etiqueta):
    """Muestra lado a lado la misma imagen antes y después de normalizar.

    Cada panel usa su propia escala de grises (0-255 y 0-1). Por eso las dos
    imágenes se ven iguales: la normalización solo cambia los valores
    numéricos, no el patrón visual del dígito.
    """
    fig, ejes = plt.subplots(1, 2, figsize=(6, 3))

    ejes[0].imshow(imagen_antes.reshape(28, 28), cmap="gray", vmin=0, vmax=255)
    ejes[0].set_title("Antes (píxeles 0-255)")
    ejes[0].axis("off")

    ejes[1].imshow(imagen_despues.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ejes[1].set_title("Después (píxeles 0-1)")
    ejes[1].axis("off")

    fig.suptitle(f"Mismo dígito ({etiqueta}) antes y después de normalizar")
    plt.tight_layout()
    plt.show()


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

    # 1) Valores originales, antes de normalizar.
    print("\nRango de píxeles antes de normalizar:")
    mostrar_rango(X_train, "X_train")
    mostrar_rango(X_test, "X_test")

    # Guardamos una imagen original para compararla después. El .copy() evita
    # que se modifique cuando normalizamos X_train.
    imagen_antes = X_train[0].copy()

    # 2) Normalizamos ambos conjuntos dividiendo entre 255.0.
    X_train, X_test = normalizar_datos(X_train, X_test)

    # 3) Comprobamos que los valores ahora estén entre 0 y 1.
    print("\nRango de píxeles después de normalizar:")
    mostrar_rango(X_train, "X_train")
    mostrar_rango(X_test, "X_test")

    # 4) Mostramos la misma imagen antes y después, lado a lado.
    comparar_normalizacion(imagen_antes, X_train[0], y_train[0])

    # 5) Finalmente, mostramos 15 ejemplos ya normalizados.
    mostrar_digitos(X_train, y_train, cantidad=15)


if __name__ == "__main__":
    main()
