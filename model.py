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
from utils import translateSentence, loadState, saveState, tokenizer, train, evaluate


df = pd.read_csv('./data/dataset_2551.csv')


### TOKENIZE and BUILD VOCABULARY
def buildVocab(sentences, tokenizer):
  counter = Counter()
  for s in sentences:
    counter.update(tokenizer(s))
  return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


requestsVocab = buildVocab(df['request'], tokenizer)
responsesVocab = buildVocab(df['response'], tokenizer)


### VECTORIZE DATA and TRAIN-VAL-TEST split it
def processData(requests, responses):
  data = []
  for (req, resp) in zip(requests, responses):
    reqTensor = torch.tensor([requestsVocab[token] for token in tokenizer(req)], dtype=torch.long)
    respTensor = torch.tensor([responsesVocab[token] for token in tokenizer(resp)], dtype=torch.long)
    data.append((reqTensor, respTensor))
  return data


trainSplit, testSplit = train_test_split(df, test_size=0.3, random_state=1)
valSplit, testSplit = train_test_split(testSplit, test_size=0.3, random_state=1)


trainData = processData(trainSplit['request'], trainSplit['response'])
valData = processData(valSplit['request'], valSplit['response'])
testData = processData(testSplit['request'], testSplit['response'])

MAX_LEN = max([len(data[1]) for data in trainData+valData+testData])
print(f"Max response len: {MAX_LEN}, requests vocab len: {len(requestsVocab)}, responses vocab len: {len(responsesVocab)}")


### BUILD DATA ITERATOR
BATCH_SIZE = 8
PAD_IDX = responsesVocab['<pad>']
BOS_IDX = responsesVocab['<bos>']
EOS_IDX = responsesVocab['<eos>']


def generateBatch(dataBatch):
  reqBatch, respBatch = [], []
  for (reqItem, respItem) in dataBatch:
    reqBatch.append(torch.cat([torch.tensor([BOS_IDX]), reqItem, torch.tensor([EOS_IDX])], dim=0))
    respBatch.append(torch.cat([torch.tensor([BOS_IDX]), respItem, torch.tensor([EOS_IDX])], dim=0))
  reqBatch = pad_sequence(reqBatch, padding_value=PAD_IDX)
  respBatch = pad_sequence(respBatch, padding_value=PAD_IDX)
  return reqBatch, respBatch


trainIter = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)
validIter = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)
testIter = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generateBatch)


### BUILD MODEL
class Transformer(nn.Module):
    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx,
        num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout,
        max_len, device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout,)
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))

        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask,)
        out = self.fc_out(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Transformer(
    embedding_size = 256,
    src_vocab_size = len(requestsVocab),
    trg_vocab_size = len(responsesVocab),
    src_pad_idx = PAD_IDX,
    num_heads = 8,
    num_encoder_layers = 3,
    num_decoder_layers = 3,
    forward_expansion = 4,
    dropout = 0.2,
    max_len = 350,
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
    #transtation = translateSentence(model, sentence1,  requestsVocab, responsesVocab, device, 350)
    #print(f"\tTranslated sentence: \n {transtation}")
 
testLoss = evaluate(model, testIter, criterion, device)
print(f'\tTest Loss: {testLoss:.4f}')


sentence1 = """GET /async/newtab_promos HTTP/1.1
Host: www.google.com
Connection: close
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: no-cors 
Sec-Fetch-Dest: empty
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9"""

sentence2 = """POST /ListAccounts?gpsia=1&source=ChromiumBrowser&json=standard HTTP/1.1
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
         
#[responsesVocab.itos[int(i)] for i in output.argmax(2).transpose(0, 1)[0]]
print(translateSentence(model, sentence1, requestsVocab, responsesVocab, device, MAX_LEN))
print(translateSentence(model, sentence2, requestsVocab, responsesVocab, device, MAX_LEN))


saveState("./save/model_2551_10_1_save.pth.tar", model, optimizer) 
loadState("./save/model_save.pth.tar", model, optimizer)
