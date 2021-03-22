# ML HTTP Fuzzer
HTTP Fuzzer backed with Machine Learning. The concept is shown on fuzzing of a server HTTP headers. More explanation to the idea can be found [here](./DOC.md).

## Installation
```
pip3 install -r requirements.txt
```
## Usage
* Newline-separated URLs in file:
```
python main.py -f ./data/test_urls.txt
```

* Single URL usage with debug output:
```
python main.py -u https://youtube.com -d
```
Debug output also contains model predictions. Example output:
![image](https://user-images.githubusercontent.com/43439351/111935941-a1135d00-8ad5-11eb-8748-d8d29d750f64.png)

## Project files
* [`./data`](./data) - contains CSV datasets and Burp output
* [`./save`](./save) - pretrained models and vocabulary
* [`main.py`](./main.py) - proof of concept
* [`fuzzer.py`](./fuzzer.py) - fuzzer implementation
* [`burpParser.py`](./burpParser.py) - used to parse Burp .XML output
* [`transformer.py`](./transformer.py) - Transformer architecture and model interface
* [`train.py`](./train.py) - preparing data and training the model
* [`utils.py`](./utils.py) - auxiliary functions
