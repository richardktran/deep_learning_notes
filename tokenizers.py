from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    sentences = [
        'I love my dog',
        'I love my cat',
        'You love my dog!',
        'Do you think my dog is amazing?'
    ]

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    print(word_index)

    new_sentences = [
        'I really love love my dog',
        'My dog loves my manatee'
    ]

    new_sequences = tokenizer.texts_to_sequences(new_sentences)

    print(new_sequences)

    # Padding to make all sequences the same length
    padding_sequences = pad_sequences(new_sequences, padding='post')

    print(padding_sequences)


if __name__ == '__main__':
    main()
