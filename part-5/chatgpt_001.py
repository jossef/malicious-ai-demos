import shutil
import tempfile
import zipfile
import openai
import dotenv
import os
import requests

dotenv.load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY


def main():
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "user", "content": "what's the capital city of Israel?"},
        ]
    )
    result = result['choices'][0]['message']['content']
    print(result)


if __name__ == '__main__':
    main()
