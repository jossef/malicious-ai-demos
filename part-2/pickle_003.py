import pickle


class MyDictionary(dict):
    def __reduce__(self):
        return list, (list(self.keys()),)


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
