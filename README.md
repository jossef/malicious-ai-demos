# Malicious AI Demos

``` 
DISCLAIMER - DO NOT RUN 
The scripts in this repository are intended solely for research purposes and should only be executed in a controlled environment. 
This experiment is designed to demonstrate potential security vulnerabilities and should not be used maliciously or for any unauthorized activities.

If you have any questions or concerns regarding this experiment, please contact us at supplychainsecurity@checkmarx.com for clarification or assistance.
By using these scripts, you agree to adhere to ethical and legal guidelines, and you accept all responsibility for any consequences that may arise from its use. 

Use it responsibly and only on systems and networks that you have explicit permission to access and assess.
```
This repository was created as part of a presentation I gave of the dangers in using modern AI tools like HuggingFace, ChatGPT and more.

- `part-1`: Usage of HuggingFace models
- `part-2`: Pickle objects and `__reduce__` function
- `part-3`: basic remote shell (client & server)
- `part-4`: creating a malicious huggingface model base on `gpt2` with the remote shell payload
- `part-5`: static code analysis using ChatGPT - can be easily manipulated

Credits:
- [Using AI to analyze if code is malicious? Think again](https://blog.illustria.io/using-ai-to-analyze-if-code-is-malicious-think-again-fb56b5f7a494)
- [The hidden dangers of loading open-source AI models (ARBITRARY CODE EXPLOIT!)](https://www.youtube.com/watch?v=2ethDz9KnLk&ab_channel=YannicKilcher)