import time


def read_data():
    with open('../data/names.txt', 'r') as f:
        words = f.read().splitlines()
        return words


def vocabulary(words):
    return sorted(list(set(''.join(words))))


def s_to_i(chars):
    return {c: i + 1 for i, c in enumerate(chars)}


def i_to_s(stoi):
    return {i: c for c, i in stoi.items()}


def training_set(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

    return xs, ys


def lesson2():
    words = read_data()
    print(words[:8])
    print(len(words))

    chars = vocabulary(words)
    print(f'vocabulary: {chars} and len is {len(chars)}')

    stoi = s_to_i(chars)
    stoi['.'] = 0
    print(stoi)

    itos = i_to_s(stoi)
    print(itos)

    print('-----training sets-----')
    xs, ys = training_set(words[:1], stoi)
    print(xs)
    print(ys)

    print('-----one hot encoding-----')
    import torch
    import torch.nn.functional as F

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    xenc = F.one_hot(xs, num_classes=27).float()
    print(xenc.shape)
    print(xenc)

    print('-----define neuron-----')
    W = torch.rand((27, 27))
    X = xenc @ W
    print(X)
    assert (X[3, 13] == (xenc[3] * W[:, 13]).sum())
    print(f'X[3, 13]={X[3, 13]} = (xenc[3] * W[:, 13]).sum()={(xenc[3] * W[:, 13]).sum()}')

    print('-----SUMMARY-----')
    g = torch.Generator().manual_seed(2147483647)
    W = torch.rand((27, 27), generator=g)

    xs, ys = training_set(words[:1], stoi)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    print('ys shape:', ys.shape)

    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    print(f'counts shape {counts.shape}')
    sum = counts.sum(1, keepdim=True)
    print(f'sum shape {sum.shape}')
    probs = counts / sum
    print(f'probs shape: {probs.shape}')

    print('---START EXPLANATION---')
    nlls = torch.zeros(5)
    for i in range(5):
        # i-th bigram
        x = xs[i].item()  # input character index
        y = ys[i].item()  # label character index
        print('----------')
        print(f'bigram example {i + 1}: {itos[x]}{itos[y]} (indexes {x},{y})')
        print(f'input to the neural net: {x} [{itos[x]}]')
        print('output probabilities from neural net:', probs[i])
        print(f'label (actual next character): {y} [{itos[y]}]')
        p = probs[i, y]
        print('probability assigned by the next to the correct character:', p.item())
        logp = torch.log(p)
        print('log likelihood:', logp.item())
        nll = -logp
        print('negative log likelihood:', nll.item())
        nlls[i] = nll

    print('=========')
    calc_loss = nlls.mean().item()
    print('average negative log likelihood, i.e. loss =', calc_loss)
    print('select indexes:', probs[torch.arange(5), ys])
    vectorized_loss = -probs[torch.arange(5), ys].log().mean()
    print('vectorized loss =', vectorized_loss)
    assert (calc_loss == vectorized_loss)
    print('---END EXPLANATION---')

    start_time = time.time()
    xs, ys = training_set(words, stoi)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    nums = xs.nelement()
    print('number of examples:', nums)

    # random weights
    g = torch.Generator().manual_seed(2147483647)
    W = torch.rand((27, 27), generator=g, requires_grad=True)

    xenc = F.one_hot(xs, num_classes=27).float()
    for k in range(100):
        # forward pass
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(nums), ys].log().mean()
        print(f'iter:{k}, loss: {loss.item()}')

        # backward pass
        W.grad = None
        loss.backward()
        W.data += -50 * W.grad

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.6f} seconds")

if __name__ == '__main__':
    lesson2()
