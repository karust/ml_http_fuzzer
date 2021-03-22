# Generative Fuzzing
## Table of Contents
1. [Introduction](#introduction)
2. [Data collection](#data-collection)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Future work](#future-work)
6. [Conclusion](#conclusion)

## Introduction
There is a possibility to adapt the text generation methods to implement a model that will produce HTTP requests/responses to fuzz whether server or client. 
This type of fuzzer can be valuable for the detection of unexpected behavior of HTTP servers/clients.

In this work, I implemented a couple of machine learning models based on **Transformer** architecture. 
These models can generate whether request or response header based on the opposite input.

Also, there is a [**PoC**](https://github.com/karust/ml_http_fuzzer/blob/main/main.py) implementation, which uses 
[the model](https://github.com/karust/ml_http_fuzzer/blob/main/save/request_model_2551_10_inf.pth.tar)
for HTTP request header generation to fuzz a server upon a given URL.

## Data collection
To achieve the task, first, I need to obtain some dataset. There is no appropriate one for this case, so I had to collected sequences of HTTP requests/responses by myself.
The obvious tool for a kind of task is **Wireshark**. Indeed, using some tweaks, it can decrypt your own TLS traffic while you surfing the Web.
There are a couple of options to save this traffic (**JSON**, **.pcap**), and several ways of parsing these files, but none of them was convenient to use.

The second option was to use **Burp Suite**. This tool also allows capturing of HTTP traffic, without any tweaks. Using Burp, it starts clean Chrome, which is good, 
since no personal data will be transmitted in the session. After that it starts to collect all the communication, which is displayed as following:

![image](https://user-images.githubusercontent.com/43439351/111936820-88a44200-8ad7-11eb-9eb3-7fc51a7ab8c6.png)

The results of sniffing can be saved in the **XML** [file](https://github.com/karust/ml_http_fuzzer/blob/main/data/burp_output_example.xml), which is very easy to [parse](https://github.com/karust/ml_http_fuzzer/blob/main/burpParser.py)
compared to Wireshark. 

*These approaches to data collection have a couple of downsides, the main one is that you have to browse the Web manually...*

## Training

## Evaluation

## Future work

## Conclusion
