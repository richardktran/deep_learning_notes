import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint

ROOT_DATASET_DIR = "./datasets"
CHECKPOINT_PATH = "models/horses_or_humans/best_model.h5"


class HorseOrHuman:
    TRAIN_DIR = ROOT_DATASET_DIR + "/horse-or-human/horse-or-human"
    VALIDATION_DIR = ROOT_DATASET_DIR + "/horse-or-human/validation-horse-or-human"
    IMAGE_WIDTH = 150
    IMAGE_HEIGHT = 150
    IMAGE_CHANNELS = 3

    def __init__(self):
        self.train_generator = ImageDataGenerator(rescale=1 / 255)
        self.validation_generator = ImageDataGenerator(rescale=1 / 255)

        self.train_data = self.train_generator.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(150, 150),
            batch_size=64,
            class_mode='binary'
        )

        self.validation_data = self.validation_generator.flow_from_directory(
            self.VALIDATION_DIR,
            target_size=(150, 150),
            batch_size=64,
            class_mode='binary'
        )

    def build_model(self):
        # CNN layers
        self.model = Sequential([
            Conv2D(16, (3, 3), activation='relu',
                   input_shape=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS)),
            MaxPool2D(2, 2),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPool2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(2, 2),
        ])

        # Dense layers
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(learning_rate=0.0001),
            metrics=['accuracy']
        )

    def load_model(self):
        self.model.load_weights(CHECKPOINT_PATH)

    def fit(self, mode="train"):
        if mode == "train":
            self.model.fit(
                self.train_data,
                epochs=15,
                validation_data=self.validation_data,
                verbose=1,
                callbacks=[self.save_checkpoint_callback()]
            )
        elif mode == "load":
            self.load_model()
        else:
            print("Invalid mode")

        return self.model

    def get_model(self) -> Sequential:
        return self.model

    def predict(self, path):
        img = image.load_img(path, target_size=(150, 150))
        image_array = image.img_to_array(img) / 255
        image_array = np.expand_dims(image_array, axis=0)

        images = np.vstack([image_array])
        classes = self.model.predict(images, batch_size=10)

        if classes[0] > 0.5:
            print(path + " is a human")
        else:
            print(path + " is a horse")

    def summary(self):
        self.model.summary()

    @staticmethod
    def save_checkpoint_callback():
        return ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            save_best_only=True,
            verbose=1
        )


if __name__ == "__main__":
    horse_or_human = HorseOrHuman()
    horse_or_human.build_model()
    # horse_or_human.summary()
    horse_or_human.fit(mode="load")
    horse_or_human.predict("./test_assets/horses_or_humans/human_1.jpeg")
