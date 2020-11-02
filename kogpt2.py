import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

tok_path = get_tokenizer()
tok_path

model,vocab = get_pytorch_kogpt2_model()

model

vocab

tok=SentencepieceTokenizer(tok_path,num_best=0,alpha=0)
tok

sent='2019년 한해를 보내며,'

# 토크나이징
toked = tok(sent)
toked

while 1:
  input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0) #unsqueeze(0)-batch추가
  pred = model(input_ids)[0]
  gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1] 
  # 제일 가능성 높은 거(argmax),axis=-1(one_hot_encoding),[-1]-맨 마지막 단어
  if gen == '</s>': # end태그나오면 끝
      break
  sent += gen.replace('▁', ' ')
  toked = tok(sent)
sent




