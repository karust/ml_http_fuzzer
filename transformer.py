import torch
import torch.nn as nn


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