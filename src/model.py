import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from preprocess import X_train, X_test, y_train, y_test, vectorizer
import pickle

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(3000,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train.toarray(),
    y_train,
    epochs=5,
    batch_size=32
)

loss, accuracy = model.evaluate(X_test.toarray(), y_test)

print("Accuracy: ", accuracy)

model.save("model.h5")
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
