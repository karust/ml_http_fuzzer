import torch
import torch.nn as nn
import pickle
import logging
from utils import translateSentence, loadState


log = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(self, embeddingSize, srcVocabSize, trgVocabSize, srcPadIdx,
        numHeads, numEncoderLayers, numDecoderLayers, forwardExpansion, dropout,
        maxLen, device,
    ):
        super(Transformer, self).__init__()
        self.srcWordEmbedding = nn.Embedding(srcVocabSize, embeddingSize)
        self.srcPositionEmbedding = nn.Embedding(maxLen, embeddingSize)
        self.trgWordEmbedding = nn.Embedding(trgVocabSize, embeddingSize)
        self.trgPositionEmbedding = nn.Embedding(maxLen, embeddingSize)

        self.device = device
        self.transformer = nn.Transformer(embeddingSize, numHeads, numEncoderLayers, numDecoderLayers, forwardExpansion, dropout,)
        self.fcOut = nn.Linear(embeddingSize, trgVocabSize)
        self.dropout = nn.Dropout(dropout)
        self.srcPadIdx = srcPadIdx


    def makeSrcMask(self, src):
        src_mask = src.transpose(0, 1) == self.srcPadIdx
        return src_mask.to(self.device)


    def forward(self, src, trg):
        srcSeqLength, N = src.shape
        trgSeqLength, N = trg.shape

        srcPositions = (torch.arange(0, srcSeqLength).unsqueeze(1).expand(srcSeqLength, N).to(self.device))
        trgPositions = (torch.arange(0, trgSeqLength).unsqueeze(1).expand(trgSeqLength, N).to(self.device))

        embedSrc = self.dropout((self.srcWordEmbedding(src) + self.srcPositionEmbedding(srcPositions)))
        embedTrg = self.dropout((self.trgWordEmbedding(trg) + self.trgPositionEmbedding(trgPositions)))

        srcPaddingMask = self.makeSrcMask(src)
        trgMask = self.transformer.generate_square_subsequent_mask(trgSeqLength).to(self.device)

        out = self.transformer(embedSrc, embedTrg, src_key_padding_mask=srcPaddingMask, tgt_mask=trgMask,)
        out = self.fcOut(out)
        return out
    

class HeaderGenerator:
    __name__ = "Transormer based HTTP header generator"

    def __init__(self, modelPath, srcVocPath, trgVocPath):
        self.sourceVocab = pickle.load(open(srcVocPath, "rb"))
        self.targetVocab = pickle.load(open(trgVocPath, "rb"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model(modelPath)
        self.model.eval()
        

    def _init_model(self, path) -> Transformer:
        model = Transformer(
            embeddingSize = 256,
            srcVocabSize = len(self.sourceVocab),
            trgVocabSize = len(self.targetVocab),
            srcPadIdx = self.targetVocab['<pad>'],
            numHeads = 8,
            numEncoderLayers = 3,
            numDecoderLayers = 3,
            forwardExpansion = 4,
            dropout = 0.2,
            maxLen = 350,
            device = self.device,
        ).to(self.device)

        log.info(f"Trying to load model on {self.device}")
        loadState(path, model, self.device)
        return model


    def translate(self, header: str) -> list:
        """Translates response headers to request or vice versa.
        Depends on the model chosen."""

        return translateSentence(self.model, header, self.sourceVocab, self.targetVocab, self.device, 350)


if __name__ == "__main__":
    tm = HeaderGenerator(
        modelPath="./save/request_model_2551_10_inf.pth.tar",
        srcVocPath="./save/srcVocab.pcl", 
        trgVocPath="./save/trgVocab.pcl"
    )

    responseHeader = """HTTP/1.1 204 No Content
Content-Type: text/html; charset=UTF-8
Date: Sat, 23 Jan 2021 17:04:43 GMT
Server: gws
Content-Length: 0
X-XSS-Protection: 0
X-Frame-Options: SAMEORIGIN
Alt-Svc: h3-29=":443"; ma=2592000,h3-T051=":443"; ma=2592000,h3-Q050=":443"; ma=2592000,h3-Q046=":443"; ma=2592000,h3-Q043=":443"; ma=2592000,quic=":443"; ma=2592000; v="46,43"
Connection: close""" 

    print(tm.translate(responseHeader))

