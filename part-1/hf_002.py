from transformers import pipeline


def main():
    generator = pipeline("text-generation")
    message = "I wish GeekDays event had a session about"
    result = generator(message, max_length=60, num_return_sequences=1)
    print(result)


if __name__ == '__main__':
    main()
