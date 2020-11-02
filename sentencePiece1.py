
import re, collections

vocab = {'l o w </w>' : 5,
         'l o w e r </w>' : 2,
         'n e w e s t </w>':6,
         'w i d e s t </w>':3
         }

def get_stats(vocab):
    # 개수를 세는 모듈
    pairs = collections.defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols)-1):
            # 연속된 두 글자씩의 빈도수 저장
            pairs[symbols[i],symbols[i+1]] += freq
            # print(pairs[symbols[i],symbols[i+1]])
    
    return pairs

get_stats(vocab)

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in v_in:
        # pair 두 글자를 붙임 (e s -> es)
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    
    return v_out

re.escape(' '.join(('e', 's')))

merge_vocab(('e', 's'), vocab)

num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)

import sentencepiece as spm

input_file = 'spm_input.txt'

corpus = [
    '딥러닝은 인공지능의 한 종류다',
    '딥러닝에는 지도학습과 비지도학습이 있다',
    '인공지능은 어렵다',
    '딥러닝도 어렵다',
    '인공지능은 신기하다'
]

with open(input_file, 'w', encoding='utf-8') as f:
    for sent in corpus:
        f.write('{}\n'.format(sent))

templates = '--input={} --model_prefix={} --vocab_size={}'

vocab_size = 30
prefix = 'data'
cmd = templates.format(input_file, prefix, vocab_size)

spm.SentencePieceTrainer.Train(cmd)

sp = spm.SentencePieceProcessor()
sp.Load('{}.model'.format(prefix))

sp.EncodeAsPieces('딥러닝은 인공지능의 한 종류다')


sp.EncodeAsIds('딥러닝은 인공지능의 한 종류다')


with open('{}.vocab'.format(prefix), encoding='utf-8') as f:
    vocabs = [doc.strip() for doc in f]

print('num of vocabs = {}'.format(len(vocabs)))


vocabs # 숫자가 낮을수록 빈도수 높은 것




