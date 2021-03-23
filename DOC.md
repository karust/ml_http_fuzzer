# Generative Fuzzing
## Table of Contents
1. [Introduction](#introduction)
2. [Data collection](#data-collection)
3. [Data analysis](#data-analysis)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Future work](#future-work)
7. [Conclusion](#conclusion)

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

In the result I collected **2551** pairs of HTTP request\responses as [CSV file](https://github.com/karust/ml_http_fuzzer/blob/10f4117ff25a7edc1e5b4a30ddd92dbd0109a8be/data/dataset_2551.csv), which were parsed from Burp XML output.

## Data analysis
Let's dissect one header from the dataset:
```
POST /ListAccounts?gpsia=1&source=ChromiumBrowser&json=standard HTTP/1.1
Host: accounts.google.com
Connection: close
Content-Length: 1
Origin: https://www.google.com
Content-Type: application/x-www-form-urlencoded
Sec-Fetch-Site: none
Sec-Fetch-Mode: no-cors
Sec-Fetch-Dest: empty
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
```

These headers easily can be represented as sentences. There are 3 main things that the model should learn to build valid headers:
* Keep `POST /ListAccounts?gpsia=1&source=ChromiumBrowser&json=standard HTTP/1.1` and similar in the beginning of header;
* Start each header from newline `\n`;
* Differentiate header name and its parameters.

As can be seen from the example of headers it consists also of numbers and "dynamically-defined" parameters. Thus, there could be a problem of enormous vocabulary if place each of these words there. The transformer model will not be able to generate any of the "dynamically-defined" parameters since its purpose here only to place words in the right order. Those dynamic params can be replaced by some placeholder and inserted after the model generates the sequence... 

The produced vocabularies on the dataset are not huge in my case: **3237** words for requests, and **7583** for responses.
The reason why responses have a bigger vocabulary is understandable if we look at the following piece of headers. They have more "dynamic parameters" in general:
```
Content-Security-Policy: script-src 'report-sample' 'nonce-I1VVO8wrGzR27/YrVdpkog' 'unsafe-inline';object-src 'none';base-uri 'self';report-uri /_/IdentityListAccountsHttp/cspreport;worker-src 'self'
Content-Security-Policy: script-src 'nonce-I1VVO8wrGzR27/YrVdpkog' 'self' 'unsafe-eval' https://apis.google.com https://ssl.gstatic.com https://www.google.com https://www.gstatic.com https://www.google-analytics.com;report-uri /_/IdentityListAccountsHttp/cspreport
Server: ESF
X-XSS-Protection: 0
Alt-Svc: h3-29=":443"; ma=2592000,h3-T051=":443"; ma=2592000,h3-Q050=":443"; ma=2592000,h3-Q046=":443"; ma=2592000,h3-Q043=":443"; ma=2592000,quic=":443"; ma=2592000; v="46,43"
```
The biggest sentence in the dataset is 342 words.

## Training
From numerous tests I decided to use the following hyperparameters for the model, using them the model is able to converge fast and is relatively compact:
```python3
model = Transformer(
    embeddingSize = 256,
    srcVocabSize = len(sourceVocab),
    trgVocabSize = len(targetVocab),
    srcPadIdx = PAD_IDX,
    numHeads = 8,
    numEncoderLayers = 3,
    numDecoderLayers = 3,
    forwardExpansion = 4,
    dropout = 0.2,
    maxLen = 350,
    device = device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

It converges to acceptable value in couple of epochs on the validation set:
![image](https://user-images.githubusercontent.com/43439351/112074249-f65b7700-8b86-11eb-8dd8-03f53db9d2b1.png)

There is no significant improvement if train model more. Obviously, we achieve this result because of the small dataset and the relative simplicity of the sequences, compared to a natural language.

## Evaluation
To evaluate the model I wrote a proof of concept application, whose algorithm looks as follows:
1. GET initial request
2. Obtain and format headers of the response
3. Pass headers to ML engine
4. Make a request with new headers
5. If the response is "similar" to the previous - finish work

As we can see from the picture below. It generates new headers as well as endpoit after the first request:
![image](https://user-images.githubusercontent.com/43439351/112078801-1d6a7680-8b90-11eb-98b3-d7e04df13e7f.png)

If look further we see that it could not generate anything new based on the next response:
![image](https://user-images.githubusercontent.com/43439351/112078866-370bbe00-8b90-11eb-8d14-0214ba309f2f.png)

It is understandable why the cycle finished so quickly - the model got the same response on the input, therefore it generated the same request for the 2nd time.
This situation also happens with most of the other URLs, though there occurred some long sequences. 

## Conclusion
After the demonstrated PoC appears a question about the suitability of the ML approach to this scenario. On the other hand, it is possible to implement a semi-ML approach to handle "dynamic-parameters". Even though the current solution cannot be counted as a valid fuzzer application, I proved that the Transformer model can be taught to generate absolutely valid HTTP headers, which can be used in a request. 

## Future work
1. Collect bigger dataset, much bigger. More different URLs.
2. For malicious activity could be used replays from different "hacking" tools, attacks.
3. Use described semi-ML approach.
4. Take into account the HTTP body - it will increase the value of the solution by a lot...
5. There is a problem when the app doesn't stop if request headers are almost identical, but some parameter changes inside which produces different lengths each time. Therefore, I could implement a better `similarity` function that will take this into account and produce a score. 
