import pandas as pd

data = pd.read_csv("data.csv", delimiter=',')

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, LSTM, Bidirectional, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam


# data split

X = data.drop('Response', axis=1)
y = data['Response']

y = to_categorical(y)

XTemp, XTest, YTemp, YTest = train_test_split(X, y, test_size=0.2, random_state=42)

XTrain, XVal, YTrain, YVal = train_test_split(XTemp, YTemp, test_size=0.2, random_state=42)

# print(XTrain.shape, XVal.shape, XTest.shape, YTest.shape, YTrain.shape, YVal.shape)

# reshaping
XTrain = XTrain.values.reshape(-1, XTrain.shape[1], 1)
XVal = XVal.values.reshape(-1, XVal.shape[1], 1)
XTest = XTest.values.reshape(-1, XTest.shape[1], 1)



# building model

model = Sequential()

model.add(Conv1D(64, 7, activation='relu', input_shape=(XTrain.shape[1], 1)))
model.add(MaxPooling1D(2))

model.add(Conv1D(32, 5, activation='relu'))
model.add(MaxPooling1D(2))




model.add(Bidirectional(LSTM(64, return_sequences=False)))


model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    XTrain,
    YTrain,
    validation_data=(XVal, YVal),
    epochs=100,
    batch_size=64
)


loss, accuracy = model.evaluate(XTest, YTest)

print(f'loss, accuracy: {loss}, {accuracy}')