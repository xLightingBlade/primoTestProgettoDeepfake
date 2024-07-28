import librosa
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_PATH = "..\\training"
JSON_PATH = "..\\data.json"
DEFAULT_SAMPLE_RATE = 22050

# valori per gli iperparametri
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 32


def prepare_dataset(dataset_path, json_path, number_of_mfcc=13, hop_length=512, n_fft=2048):
    data = {
        "mappings": [],
        "labels": [],
        "mfcc": [],
        # "mel_spectrogram": [],
        "file_path": []
    }
    for root, dirs, files in os.walk(dataset_path):
        if root is not dataset_path:
            label_name = root.split("\\")[-1]  #l'output è una lista ["training", "fake"] e poi ["training","real"]
            #avendo solo due categorie si poteva magari fare senza impostare questo ciclo, però manteniamo flessibiltà
            print(f"Processing {label_name} audio files")
            data["mappings"].append(label_name)
            for file in files:
                file_path = os.path.join(root, file)
                print(f"file path : {file_path}")
                audio_signal, sample_rate = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)
                print(audio_signal.shape)
                #i file audio di questo dataset sono tutti troncati a 2 secondi quindi hanno la stessa forma

                #coefficienti (mfcc) della traccia audio
                mfcc_list = librosa.feature.mfcc(y=audio_signal, n_mfcc=number_of_mfcc,
                                                 n_fft=n_fft, hop_length=hop_length)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=DEFAULT_SAMPLE_RATE,
                                                                 n_fft=n_fft, hop_length=hop_length)

                label = 0 if label_name == 'real' else 1
                data["labels"].append(label)
                #librosa restituisce gli mfcc come un array 2d, noi vorremmo 1d quindi facciamo la trasposta
                data["mfcc"].append(mfcc_list.T.tolist())
                #provo ad estrarre anche gli spettrogrammi mel-scaled
                #data["mel_spectrogram"].append(mel_spectrogram.T.tolist())
                data["file_path"].append(file_path)
                print(f"{file_path} is labeled {label_name}({label})")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def load_input_and_target_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y


def build_conv_model_demo(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    model.summary()

    return model


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    text = f"Learning rate of {LEARNING_RATE}, trained for {EPOCHS} epochs"
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()


def main():

    # prepare_dataset(DATASET_PATH, JSON_PATH)  #commentato, già eseguito una volta. TODO refactor
    x, y = load_input_and_target_data(JSON_PATH)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(X_train.shape)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)  #(numsegmenti, 13, 1)
    model = build_conv_model_demo(input_shape, learning_rate=LEARNING_RATE)
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=BATCH_SIZE)
    plot_history(history)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, test_acc))

    # save model
    model.save("..\\model.keras")


if __name__ == "__main__":
    main()
