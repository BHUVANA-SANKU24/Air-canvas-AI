import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = "data/quickdraw"
CLASSES = ["car", "tree", "house", "cat", "bicycle"]
SAMPLES_PER_CLASS = 20000  # increase for better accuracy

X_list, y_list = [], []

for i, cls in enumerate(CLASSES):
    path = os.path.join(DATA_DIR, f"{cls}.npy")
    data = np.load(path)[:SAMPLES_PER_CLASS]
    data = data.reshape(-1, 28, 28)
    X_list.append(data)
    y_list.append(np.full((data.shape[0],), i))

X = np.concatenate(X_list, axis=0).astype("float32") / 255.0
X = X.reshape(-1, 28, 28, 1)

y = np.concatenate(y_list, axis=0)
y = to_categorical(y, num_classes=len(CLASSES))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(len(CLASSES), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

early = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(" Test accuracy:", acc)

os.makedirs("models", exist_ok=True)
model.save("models/quickdraw_model.h5")
print(" Saved model to models/quickdraw_model.h5")
