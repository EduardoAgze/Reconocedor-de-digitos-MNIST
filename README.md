# Reconocedor Neuronal de Dígitos Manuscritos

Carga, preprocesa y visualiza el dataset **MNIST** con `scikit-learn` y `matplotlib`. El proyecto sienta las bases para entrenar una red neuronal capaz de reconocer dígitos escritos a mano.

## ✨ Qué hace

- Descarga MNIST con `fetch_openml` (70 000 imágenes de 28×28 píxeles).
- Separa **60 000** ejemplos para entrenamiento y **10 000** para prueba.
- Muestra las formas de `X_train`, `y_train`, `X_test` y `y_test`.
- Normaliza los píxeles al rango `[0, 1]` dividiendo entre `255.0`.
- Compara una misma imagen antes y después de normalizar.
- Visualiza una grilla con 15 dígitos de ejemplo.

## 📦 Requisitos

- Python 3.10 o superior
- `numpy`, `matplotlib`, `scikit-learn` y `pandas<3`

## 🚀 Instalación y uso

```bash
# Crear y activar el entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el script
python src/cargar_mnist.py
```

> La primera ejecución descarga MNIST (~11 MB). Las siguientes usan la copia en caché.

## 📊 Salida esperada

```text
Formas de los arrays:
  X_train: (60000, 784)
  y_train: (60000,)
  X_test:  (10000, 784)
  y_test:  (10000,)

Rango de píxeles antes de normalizar:
  X_train: mínimo = 0.0, máximo = 255.0

Rango de píxeles después de normalizar:
  X_train: mínimo = 0.0, máximo = 1.0
```

Además se abren dos ventanas de matplotlib: la comparación antes/después y la grilla de 15 dígitos.

## 📁 Estructura

```text
.
├── src/
│   └── cargar_mnist.py
├── requirements.txt
├── .gitignore
└── README.md
```
