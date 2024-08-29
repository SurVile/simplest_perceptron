import telebot
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    bot = telebot.TeleBot('7256498859:AAFtuIPbvcfhuR0Sawr3_BKo1CySy9AYBKw')

    # Загружаем модель
    model = tf.keras.models.load_model('my_model.h5')

    # Загружаем TfidfVectorizer и LabelEncoder
    vectorizer = joblib.load('vectorizer.pkl')
    encoder_answers = joblib.load('encoder_answers.pkl')

    @bot.message_handler(commands=['start'])
    def say_hi(message):
        bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}!')

    @bot.message_handler(commands=['ai'])
    def start_ai(message):
        bot.send_message(message.chat.id, 'Что ты хочешь узнать?')

        @bot.message_handler()
        def ai_answer(message):
            question = [message.text]
            question_vectorized = vectorizer.transform(question).toarray()

            # Вычисляем значения
            probs = model.predict(question_vectorized)

            # Вычисляем наибольшую вероятность
            pred_class = np.argmax(probs, axis=1)

            # Создаем список ответов
            answers = ["Синего", "Зеленого", "Красного", "Синего", "Зеленого", "Желтого"]

            bot.send_message(message.chat.id, f'Твой ответ: {answers[pred_class[0]]}')

    bot.polling(none_stop=True)

if __name__ == '__main__':
    main()
