import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_hub import KerasLayer
from tensorflow.keras.callbacks import ModelCheckpoint

DATASET = "horse_or_human"

ROOT_DATASET_DIR = "./datasets/{}".format(DATASET)
ROOT_TEST_ASSETS_DIR = "./test_assets/{}/".format(DATASET)
SAVED_MODEL_PATH = "models/{}/resnet_model.h5".format(DATASET)


class ResnetHorseOrHuman:
    TRAIN_DIR = ROOT_DATASET_DIR + "/horse-or-human"
    VALIDATION_DIR = ROOT_DATASET_DIR + "/validation-horse-or-human"
    RESNET_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
    IMAGE_SIZE = (224, 224)

    def __init__(self):
        self.train_generator = ImageDataGenerator(rescale=1 / 255)
        self.validation_generator = ImageDataGenerator(rescale=1 / 255)

        self.train_data = self.train_generator.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(224, 224),
            batch_size=64
        )

        self.validation_data = self.validation_generator.flow_from_directory(
            self.VALIDATION_DIR,
            target_size=(224, 224),
            batch_size=64
        )
        self.num_of_classes = len(self.train_data.class_indices)

    def build_model(self) -> Sequential:
        self.model = Sequential([
            KerasLayer(self.RESNET_URL, trainable=False, input_shape=(*self.IMAGE_SIZE, 3)),
            Dense(self.num_of_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def fit(self, epochs=15, mode='train') -> Sequential:
        if mode == 'train':
            self.model.fit(
                self.train_data,
                epochs=epochs,
                validation_data=self.validation_data,
                verbose=1,
                callbacks=[self.save_model()]
            )
        elif mode == 'load':
            self.model.load_weights(SAVED_MODEL_PATH)
        else:
            raise Exception("Invalid mode")

        return self.model

    @staticmethod
    def save_model():
        return ModelCheckpoint(
            filepath=SAVED_MODEL_PATH,
            save_best_only=True,
            verbose=1
        )

    def predict(self, image_path) -> str:
        image_path = ROOT_TEST_ASSETS_DIR + image_path
        img = load_img(image_path, target_size=self.IMAGE_SIZE)
        img = img_to_array(img) / 255
        img = np.expand_dims(img, axis=0)
        images = np.vstack([img])

        classes = self.model.predict(images, batch_size=10)

        # Invert classes indices
        classes_indices = {v: k for k, v in self.train_data.class_indices.items()}

        return classes_indices[np.argmax(classes)]

    def summary(self):
        self.model.summary()


if __name__ == "__main__":
    horse_or_human = ResnetHorseOrHuman()
    horse_or_human.build_model()
    horse_or_human.summary()
    horse_or_human.fit(epochs=15, mode='load')

    print(horse_or_human.predict("horse_cartoon_1.jpg"))
