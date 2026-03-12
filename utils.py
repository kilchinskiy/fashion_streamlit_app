# Імпортуємо TensorFlow для роботи з нейронними мережами
import tensorflow as tf

# Імпортуємо numpy для роботи з масивами
import numpy as np

# Імпортуємо PIL для роботи із зображеннями
from PIL import Image


# Список назв класів датасету Fashion MNIST
class_names = [
'T-shirt/top',
'Trouser',
'Pullover',
'Dress',
'Coat',
'Sandal',
'Shirt',
'Sneaker',
'Bag',
'Ankle boot'
]



# Функція завантаження навченої CNN моделі
def load_cnn():

    # Завантажуємо модель з файлу model_cnn.h5
    # compile=False означає, що нам не потрібно заново компілювати модель
    return tf.keras.models.load_model("model_cnn.h5", compile=False)




# Функція створення моделі на основі VGG16
def load_vgg():

    # Завантажуємо базову модель VGG16 з попередньо навченими вагами ImageNet
    base_model = tf.keras.applications.VGG16(
        weights="imagenet",     # використовуємо ваги ImageNet
        include_top=False,      # не використовуємо верхній класифікаційний шар
        input_shape=(96,96,3)   # розмір вхідного зображення
    )

    # Заморожуємо шари базової моделі
    base_model.trainable = False

    # Створюємо нову модель на основі VGG16
    model = tf.keras.Sequential([

        # Базова згорткова частина VGG16
        base_model,

        # Перетворюємо карту ознак у вектор
        tf.keras.layers.GlobalAveragePooling2D(),

        # Повнозв'язний шар
        tf.keras.layers.Dense(256, activation="relu"),

        # Вихідний шар для 10 класів
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Повертаємо створену модель
    return model




# Функція попередньої обробки зображення для CNN
def preprocess_cnn(image):

    # Змінюємо розмір зображення до 28x28
    image = image.resize((28,28))

    # Перетворюємо зображення у numpy масив
    image = np.array(image)

    # Якщо зображення RGB – беремо тільки один канал
    if len(image.shape) == 3:
        image = image[:,:,0]

    # Нормалізуємо значення пікселів (0–255 → 0–1)
    image = image / 255.0

    # Додаємо вимір каналу
    image = np.expand_dims(image, axis=-1)

    # Додаємо batch dimension
    image = np.expand_dims(image, axis=0)

    # Повертаємо оброблене зображення
    return image




# Функція попередньої обробки для VGG16
def preprocess_vgg(image):

    # Змінюємо розмір до 96x96
    image = image.resize((96,96))

    # Перетворюємо у numpy масив
    image = np.array(image)

    # Якщо зображення grayscale – робимо RGB
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    # Нормалізуємо значення пікселів
    image = image / 255.0

    # Додаємо batch dimension
    image = np.expand_dims(image, axis=0)

    # Повертаємо оброблене зображення
    return image




# Функція передбачення класу
def predict(model, image, model_type):

    # Якщо використовується CNN
    if model_type == "CNN":

        # Виконуємо preprocessing для CNN
        img = preprocess_cnn(image)

    else:

        # Виконуємо preprocessing для VGG16
        img = preprocess_vgg(image)

    # Робимо передбачення
    preds = model.predict(img)

    # Отримуємо ймовірності
    probs = preds[0]

    # Визначаємо клас з найбільшою ймовірністю
    predicted_class = class_names[np.argmax(probs)]

    # Повертаємо probabilities та predicted class
    return probs, predicted_class