# tokenizer

A simple and efficient tokenizer for natural language processing tasks. This project includes both a basic byte-level Byte Pair Encoding (BPE) tokenizer and a regex-based tokenizer.

## Installation

To install the tokenizer, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/tokenizer.git
cd tokenizer
pip install -r requirements.txt
```

## Usage

Here is an example of how to use the tokenizer:

```python
from src.basic import BasicTokenizer
from src.regexTokenizer import RegexTokenizer

# Load your text data
text = "Your text data goes here."

# Initialize the BasicTokenizer
basic_tokenizer = BasicTokenizer()
basic_tokenizer.train(text, vocab_size=512, verbose=True)
tokens = basic_tokenizer.encode(text)
decoded_text = basic_tokenizer.decode(tokens)

# Initialize the RegexTokenizer
regex_tokenizer = RegexTokenizer()
regex_tokenizer.train(text, vocab_size=512, verbose=True)
tokens = regex_tokenizer.encode(text)
decoded_text = regex_tokenizer.decode(tokens)
```

## Training

To train the tokenizers on some data, you can use the provided `train.py` script:

```bash
python train.py
```

This script will train both the `BasicTokenizer` and `RegexTokenizer` on the text data located in `data/taylorswift.txt` and save the models in the `models` directory.

## References

This project is inspired by the [minBPE](https://github.com/karpathy/minBPE) repository by Karpathy.

## License

This project is licensed under the MIT License.