from transformers import pipeline


def main():
    classifier = pipeline("sentiment-analysis")
    message = "The GeekDays event is awesome!"
    # message = "The GeekDays event is not what I expected..."
    result = classifier(message)
    print(result)


if __name__ == '__main__':
    main()
