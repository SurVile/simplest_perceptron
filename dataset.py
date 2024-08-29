import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Пример создания DataFrame
data = {
    "Вопрос": ["Какого цвета небо?", "Какого цвета трава?", "Какого цвета огонь?",
               "Какого цвета море?", "Какого цвета листья?", "Какого цвета солнце?"],
    "Ответ": ["Синего", "Зеленого", "Красного", "Синего", "Зеленого", "Желтого"]
}
df = pd.DataFrame(data)

# Преобразуем вопросы (категориальные признаки) в числовые с помощью TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Вопрос"]).toarray()

# Преобразуем ответы (целевую переменную) в числовые значения
encoder_answers = LabelEncoder()
y = encoder_answers.fit_transform(df["Ответ"])

# Преобразуем признаки и целевую переменную в массивы NumPy
features = X
targets = y.reshape(-1, 1)

# Разделяем данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Создаем dataset TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Преобразуем dataset в батчи
batch_size = 2
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Создаем модель нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(encoder_answers.classes_), activation='softmax')  # Количество классов (ответов)
])

# Компилируем модель
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
model.fit(train_dataset, epochs=1500, validation_data=test_dataset)

# Оцениваем модель
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Сохраняем модель
model.save('my_model.h5')

# Сохраняем TfidfVectorizer и LabelEncoder
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(encoder_answers, 'encoder_answers.pkl')