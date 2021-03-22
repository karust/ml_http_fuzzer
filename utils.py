import torch


def tokenizer(sentence):
    """Split header text to tokens, add newline symbols"""
    return " \n ".join(sentence.split("\n")).split(" ")


# "".join([requestsVocab.itos[i] for i in source.transpose(0, 1)[1]])
#"".join([responsesVocab.itos[int(i)] for i in trainOutput.argmax(2).transpose(0, 1)[0]])
def train(model, iterator, optimizer, criterion, device):
    """Train model and calculate mean loss"""
    
    model.train()
    losses = []
    
    for _, (source, target) in enumerate(iterator):
        # Get input and targets and get to cuda
        source, target = source.to(device), target.to(device)

        # Forward prop
        output = model(source, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())
        
        # Back prop
        loss.backward()
        
        # Clip to avoid exploding gradient issues, makes sure grads are within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()
    # Return mean loss
    return sum(losses) / len(losses)
        

def evaluate(model, iterator, criterion, device):
    """Evaluate model and calculate mean loss"""
     
    model.eval()
    losses = []

    with torch.no_grad():
        for _, (source, target) in enumerate(iterator):
            source, target = source.to(device), target.to(device)

            output = model(source, target[:-1, :])
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            loss = criterion(output, target)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def epochTime(startTime: int, endTime: int):
    """Calculate elapsed (elapsedMins, elapsedSecs) of epoch"""
     
    elapsedTime = endTime - startTime
    elapsedMins = int(elapsedTime / 60)
    elapsedSecs = int(elapsedTime - (elapsedMins * 60))
    return elapsedMins, elapsedSecs


def translateSentence(model, sentence: str, srcLang, trgLang, device, maxLength):
    """Translate header request to response or vice versa"""
    
    tokens = [token for token in tokenizer(sentence)]
    textIndices = [srcLang.stoi[token] for token in tokens]
    textIndices.insert(0, srcLang.stoi["<bos>"])
    textIndices.append(srcLang.stoi["<eos>"])
    
    sentenceTensor = torch.LongTensor(textIndices).unsqueeze(1).to(device)

    outputs = [trgLang.stoi["<bos>"]]
    
    for i in range(maxLength):
        trgTensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        
        with torch.no_grad():
            output = model(sentenceTensor, trgTensor)
        
        guess = output.argmax(2)[-1, :].item()     
        outputs.append(guess)
        
        if guess == trgLang.stoi["<eos>"]:
           break
    
    return [trgLang.itos[idx] for idx in outputs]


def loadState(filename, model, device='cpu', optimizer=None):
    checkpoint = torch.load(filename, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        model.eval()
    
    
def saveState(filename, model, optimizer=None):
    torch.save({"model": model.state_dict(), 
                "optimizer": optimizer.state_dict() if optimizer else None}, 
                filename)
    