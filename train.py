import time
from collections import Counter
import pandas as pd
import torch
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformer import Transformer
from utils import *


df = pd.read_csv('./data/dataset_2551.csv')

# Define what to generate - responses or requests
target, source = "request", "response" 


### TOKENIZE and BUILD VOCABULARY
def buildVocab(sentences, tokenizer):
  counter = Counter()
  for s in sentences:
    counter.update(tokenizer(s))
  return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


sourceVocab = buildVocab(df[source], tokenizer)
targetVocab = buildVocab(df[target], tokenizer)


### VECTORIZE DATA and TRAIN-VAL-TEST split it
def processData(sourceData, targetData):
  data = []
  for (source, target) in zip(sourceData, targetData):
    reqTensor = torch.tensor([sourceVocab[token] for token in tokenizer(source)], dtype=torch.long)
    respTensor = torch.tensor([targetVocab[token] for token in tokenizer(target)], dtype=torch.long)
    data.append((reqTensor, respTensor))
  return data

# Train-Test-Validation split
trainSplit, testSplit = train_test_split(df, test_size=0.3, random_state=1)
valSplit, testSplit = train_test_split(testSplit, test_size=0.3, random_state=1)

trainData = processData(trainSplit[source], trainSplit[target])
valData = processData(valSplit[source], valSplit[target])
testData = processData(testSplit[source], testSplit[target])

MAX_LEN = max([max(len(data[0]), len(data[1])) for data in trainData+valData+testData])
print(f"Max request/response len: {MAX_LEN}, requests vocab len: {len(sourceVocab)}, responses vocab len: {len(targetVocab)}")


### BUILD DATA ITERATOR
BATCH_SIZE = 8
PAD_IDX = targetVocab['<pad>']
BOS_IDX = targetVocab['<bos>']
EOS_IDX = targetVocab['<eos>']

def generateBatch(dataBatch):
  srcBatch, targBatch = [], []
  for (reqItem, respItem) in dataBatch:
    srcBatch.append(torch.cat([torch.tensor([BOS_IDX]), reqItem, torch.tensor([EOS_IDX])], dim=0))
    targBatch.append(torch.cat([torch.tensor([BOS_IDX]), respItem, torch.tensor([EOS_IDX])], dim=0))
  srcBatch = pad_sequence(srcBatch, padding_value=PAD_IDX)
  targBatch = pad_sequence(targBatch, padding_value=PAD_IDX)
  return srcBatch, targBatch

trainIter = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)
validIter = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)
testIter = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)


### BUILD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


### TRAIN AND EVALUATE
for epoch in range(10):
    startTime = time.time()
       
    trainLoss = train(model, trainIter, optimizer, criterion, device)
    validLoss = evaluate(model, validIter, criterion, device)
    
    scheduler.step(trainLoss)
    
    endTime = time.time()
    epochMins, epochSecs = epochTime(startTime, endTime)  
    print(f'Epoch: {epoch+1} | Time: {epochMins}m {epochSecs}s')
    print(f'\tTrain Loss: {trainLoss:.4f}')
    print(f'\tVal Loss: {validLoss:.4f}')
    #transtation = translateSentence(model, sentence1,  sourceVocab, targetVocab, device, 350)
    #print(f"\tTranslated sentence: \n {transtation}")
 
testLoss = evaluate(model, testIter, criterion, device)
print(f'\tTest Loss: {testLoss:.4f}')


request1 = """GET /async/newtab_promos HTTP/1.1
Host: www.google.com
Connection: close
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: no-cors 
Sec-Fetch-Dest: empty
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9"""

request2 = """POST /ListAccounts?gpsia=1&source=ChromiumBrowser&json=standard HTTP/1.1
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
Accept-Language: en-US,en;q=0.9"""

#[targetVocab.itos[int(i)] for i in output.argmax(2).transpose(0, 1)[0]]
print(translateSentence(model, request1, sourceVocab, targetVocab, device, MAX_LEN))
print(translateSentence(model, request2, sourceVocab, targetVocab, device, MAX_LEN))

response1 = """HTTP/1.1 200 OK
Version: 352753590
Content-Type: application/json; charset=UTF-8
X-Content-Type-Options: nosniff
Content-Disposition: attachment; filename="f.txt"
Date: Sat, 23 Jan 2021 17:04:36 GMT
Server: gws
Cache-Control: private
X-XSS-Protection: 0
X-Frame-Options: SAMEORIGIN
Alt-Svc: h3-29=":443"; ma=2592000,h3-T051=":443"; ma=2592000,h3-Q050=":443"; ma=2592000,h3-Q046=":443"; ma=2592000,h3-Q043=":443"; ma=2592000,quic=":443"; ma=2592000; v="46,43"
Connection: close
Content-Length: 29"""

response2 = """HTTP/1.1 204 No Content
Content-Type: text/html; charset=UTF-8
Date: Sat, 23 Jan 2021 17:04:43 GMT
Server: gws
Content-Length: 0
X-XSS-Protection: 0
X-Frame-Options: SAMEORIGIN
Alt-Svc: h3-29=":443"; ma=2592000,h3-T051=":443"; ma=2592000,h3-Q050=":443"; ma=2592000,h3-Q046=":443"; ma=2592000,h3-Q043=":443"; ma=2592000,quic=":443"; ma=2592000; v="46,43"
Connection: close"""  
   
#[targetVocab.itos[int(i)] for i in output.argmax(2).transpose(0, 1)[0]]
print(translateSentence(model, response1, sourceVocab, targetVocab, device, MAX_LEN))
print(translateSentence(model, response2, sourceVocab, targetVocab, device, MAX_LEN))

saveState("./save/request_model_2551_10_inf.pth.tar", model) 
loadState("./save/response_model_2551_10_inf.pth.tar", model, optimizer)


import pickle
with open("./save/srcVocab.pcl", 'wb') as output:
    pickle.dump(sourceVocab, output)

with open("./save/trgVocab.pcl", 'wb') as output:
    pickle.dump(targetVocab, output)