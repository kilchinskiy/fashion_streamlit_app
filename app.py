# Імпортуємо Streamlit для створення веб-застосунку
import streamlit as st

# Імпортуємо pickle для завантаження history моделей
import pickle

# Імпортуємо matplotlib для побудови графіків
import matplotlib.pyplot as plt

# Імпортуємо бібліотеку для роботи із зображеннями
from PIL import Image

# Імпортуємо функції з utils.py
from utils import load_cnn, load_vgg, predict, class_names


# Заголовок веб-застосунку
st.title("Fashion MNIST Image Classifier")


# Короткий опис застосунку
st.write("Upload an image and choose a model")



# Вибір моделі
# Створюємо dropdown список для вибору моделі
model_option = st.selectbox(
    "Choose model",
    ["CNN", "VGG16"]
)




# Завантаження моделі
# Використовуємо кешування Streamlit
# щоб модель не завантажувалась кожного разу
@st.cache_resource
def get_model(name):

    # Якщо вибрана CNN модель
    if name == "CNN":

        # Завантажуємо CNN
        return load_cnn()

    # Інакше використовуємо VGG16
    else:

        # Створюємо модель VGG16
        return load_vgg()


# Завантажуємо модель
model = get_model(model_option)




# Завантаження history для побудови графіків
# Якщо вибрана CNN модель
if model_option == "CNN":

    # Використовуємо history CNN
    history_file = "history_cnn.pkl"

else:

    # Використовуємо history VGG16
    history_file = "history_vgg16.pkl"


# Відкриваємо файл history
with open(history_file, "rb") as f:

    # Завантажуємо history
    history = pickle.load(f)




# Побудова графіків навчання
# Заголовок розділу
st.subheader("Training graphs")

# Створюємо фігуру з двома графіками
fig, ax = plt.subplots(1,2)


# Графік функції втрат
ax[0].plot(history["loss"], label="train")
ax[0].plot(history["val_loss"], label="validation")
ax[0].set_title("Loss")
ax[0].legend()


# Графік точності
ax[1].plot(history["accuracy"], label="train")
ax[1].plot(history["val_accuracy"], label="validation")
ax[1].set_title("Accuracy")
ax[1].legend()


# Відображаємо графіки у Streamlit
st.pyplot(fig)




# Завантаження зображення
# Створюємо кнопку upload
uploaded_file = st.file_uploader(
    "Upload clothing image",
    type=["jpg","png","jpeg"]
)




# Якщо користувач завантажив зображення

if uploaded_file:

    # Відкриваємо зображення
    image = Image.open(uploaded_file)

    # Відображаємо зображення
    st.image(image, caption="Uploaded image")


    # Робимо передбачення
    probs, pred = predict(model, image, model_option)


    # Виводимо заголовок
    st.subheader("Prediction")


    # Показуємо передбачений клас
    st.success(f"Predicted class: {pred}")


    # Заголовок ймовірностей
    st.subheader("Class probabilities")


    # Виводимо ймовірності для кожного класу
    for i,p in enumerate(probs):

        st.write(f"{class_names[i]} : {p:.3f}")


    # Відображаємо графік ймовірностей
    st.bar_chart(probs)