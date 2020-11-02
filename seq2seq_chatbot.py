from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import re

from konlpy.tag import Okt


# # 데이터 로드
# 디코더 입력에 START가 들어가면 디코딩의 시작 의미. 반대로 디코더 출력에 END가 나오면 디코딩 종료

# 태그 단어
PAD = "<PADDING>"
STA = "<STA>"
END = "<END>"
OOV = "<OOV>"

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입
ENCODER_INPUT = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 30

# 임베딩 벡터 차원
embedding_dim = 100

# LSTM 히든 레이어 차원
lstm_hidden_dim = 128

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()]")

# 챗봇 데이터 로드
chatbot_data = pd.read_csv(r'D:\강의자료\4 - 딥러닝 자연어처리\8 - Seq2Seq의 응용\실습\dataset\chatbot\ChatbotData.csv',encoding='utf-8')
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])


len(question)


# 데이터의 일부만 학습에 사용
question = question[:100]
answer = answer[:100]

# 챗봇 데이터 출력
for i in range(10):
    print('Q:'+question[i])
    print('A:'+answer[i])
    print()
# # 단어 사전 생성

# 형태소 분석 함수 
def pos_tag(sentences):
    # KoNLPy 형태소 분석기 설정
    tagger = Okt()
    # 문장 품사 변수 초기화
    sentences_pos = []
    # 모든 문장 반복
    for sentence in sentences:
        # 특수기호 제거
        sentence = re.sub(RE_FILTER,"",sentence)

        # 배열인 형태소분석의 출력을 띄어쓰기로 구분하여 붙임
        # 형태소 단위로 문자열을 끊고 싶다면, .morphs()를 사용하면 된다
        sentence = " ".join(tagger.morphs(sentence))
        sentences_pos.append(sentence)
    return sentences_pos


# 형태소 분석 수행
question = pos_tag(question)
answer = pos_tag(answer)
print(question)
print(answer)
# 형태소 분석으로 변환된 챗봇 데이터 출력
for i in range(10):
    print('Q:' + question[i])
    print('A:' + answer[i])
    print()


# 질문과 대답 문장들을 하나로 합침
sentences=[]
sentences.extend(question)
sentences.extend(answer)

words=[]

# 단어들의 배열 생성
for sentence in sentences:
    for word in sentence.split():
        words.append(word)
# 길이가 0인 단어는 삭제
words = [word for word in words if len(word)>0]

# 중복된 단어 삭제
words = list(set(words))

# 제일 앞에 태그 단어 삽입
words[:0] = [PAD, STA, END, OOV]
# 질문과 대답 문장들을 합쳐서 전체 단어사전 만들기. 자연어 처리에서는 항상 이렇게 단어를 인덱스에 따라서 정리.그래야지 문장을 인덱스 배열로 바꿔서 임베딩 레이어에 넣을 수 있습니다. 
# 또한 모델의 출력에서 나온 인덱스를 다시 단어로 변환하는데도 필요

# 단어 개수
len(words)


# 단어 출력
words


# 단어와 인덱스의 딕셔너리 생성
word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}


# 단어->인덱스
# 문장을 인덱스로 변환하여 모델 입력으로 사용
word_to_index


# 인덱스 -> 단어
# 문장을 인덱스로 변환하여 모델 입력으로 사용
index_to_word
# # 전처리

# 문장을 인덱스로 변환
def convert_text_to_index(sentences, vocabulary, type):
    sentences_index = []

    # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = []

        # 디코더 입력일 경우 맨 앞에 START태그 추가
        if type==DECODER_INPUT:
            sentence_index.extend([vocabulary[STA]])

        # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence.split():
            if vocabulary.get(word) is not None:
                #사전에 있는 단어면 해당 인덱스를 추가
                sentence_index.extend([vocabulary[word]])
            else:
                #사전에 없는 단어면 OOV인덱스 추가
                sentence_index.extend([vocabulary[OOV]])
        # 최대 길이 검사
        if type == DECODER_TARGET:
            # 디코더 목표일 경우 맨 뒤에 END태그 추가
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else:
                sentence_index+=[vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]
        
        # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index+=(max_sequences-len(sentence_index))*[vocabulary[PAD]]

        # 문장의 인덱스 배열을 추가
        sentences_index.append(sentence_index)

    return np.asarray(sentences_index)
    

# 원래 seq2seq는 디코더의 현재 출력이 디코더의 다음 입력으로 들어갑니다. 다만 학습에서는 굳이 이렇게 하지 않고 디코더 입력과 디코더 출력의 데이터를 각각 만듭니다.
# 
# 그러나 예측시에는 이런 방식이 불가능. 출력값을 미리 알지 못하기 때문에, 디코더 입력을 사전에 생성할 수가 없습니다. 이런 문제를 해결하기 위해 훈련 모델과 예측 모델을 따로 구성해야 합니다.

# 인코더 입력 인덱스 변환
x_encoder = convert_text_to_index(question, word_to_index, ENCODER_INPUT)

# 첫번째 인코더 입력 출력(12시 땡)
x_encoder[0]


# 디코더 입력 인덱스 변환
x_decoder = convert_text_to_index(answer, word_to_index, DECODER_INPUT)

# 첫번째 디코더 입력 출력(START 하루 가 또 가네요)
x_decoder[0]


# 디코더 목표 인덱스 변환
y_decoder = convert_text_to_index(answer,word_to_index, DECODER_TARGET)

# 첫 번재 디코더 목표 출력(하루 가 또 가네요 END)
y_decoder[0] # 2가 end tag
len(y_decoder)


# 원핫인코딩 초기화
one_hot_data = np.zeros((len(y_decoder), max_sequences, len(words))) #(100,30,454)===>(batch, 문장 길이, 단어개수)

# 디코더 목표를 원핫인코딩으로 변환 #DECODER_TARGET
# 학습 시 입력은 인덱스이지만, 출력은 원핫인코딩 형식
for i, sequence in enumerate(y_decoder):
    for j, index in enumerate(sequence):
        one_hot_data[i,j,index]=1

# 디코더 목표 설정
y_decoder = one_hot_data

# 첫번째 디코더 목표 출력
y_decoder[0]
print(y_decoder[0].shape) #(30,454)
print(y_decoder.shape)    #(100,30,454) (30,454)가 총 100개 있음

# 인코더 입력과 디코더 입력은 임베딩 레이어에 들어가는 인덱스 배열입니다. 반면에 디코더 출력은 원핫인코딩 형식이어야 합니다. 디코더의 마지막 Dense레이어에서 Softmax로 나오기 때문 [markdown]
# # 모델 생성

#-----------------------------------
# 훈련 모델 인코더 정의
#-----------------------------------

# 입력 문장의 인덱스 시퀀스를 입력으로 받음
encoder_inputs = layers.Input(shape=(None,),name='A1')    

# 임베딩 레이어
encoder_outputs = layers.Embedding(len(words), embedding_dim,name='A2')(encoder_inputs) #(454,100) #[(None,None,100)]

# return_state가 True면 상태값 리턴
# LSTM은 state_h와 state_c 2개의 상태 존재
# recurrent_dropout은 state 삭제시키는 것
encoder_outputs, state_h, state_c = layers.LSTM(lstm_hidden_dim, dropout=0.1, recurrent_dropout=0.5, return_state=True,name='A5')(encoder_outputs) #[(None,128),None// parameters: 117,248(4*128((128+1(bias)+100(embedding_dim)))

# 히든 상태와 셀 상태를 하나로 묶음
encoder_states = [state_h, state_c]

#-------------------------------------
# 훈련 모델 디코더 정의
#------------------------------------

# 목표 문장의 인덱스 시퀀스를 입력으로 받음
decoder_inputs = layers.Input(shape=(None,),name='A3')

# 임베딩 레이어
decoder_embedding = layers.Embedding(len(words),embedding_dim,name='A4') #(454,100)
decoder_outputs = decoder_embedding(decoder_inputs)
print(decoder_outputs)

# 인코더와 달리 return_sequences를 True로 설정하여 모든 타임스텝 출력값 리턴
# 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리가히 위함
decoder_lstm  = layers.LSTM(lstm_hidden_dim, dropout=0.1, recurrent_dropout=0.5, return_state=True, return_sequences=True,name='A6')


# initial_state를 인코더의 상태로 초기화
decoder_outputs,_,_=decoder_lstm(decoder_outputs, initial_state=encoder_states)

# 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
decoder_dense = layers.Dense(len(words),activation='softmax',name='A7')
decoder_outputs = decoder_dense(decoder_outputs)

#-------------------------------------------
# 훈련 모델 정의
#-------------------------------------------

# 입력과 출력으로 함수형 API모델 생성
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 학습 방법 설정
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
model.summary()
 [markdown]
# 지금까지의 예제는 Sequential방식의 모델이었습니다. 이번에는 함수형 API모델 사용. 인코더와 디코더가 따로 분리되어야 하는데, 단순히 레이어를 추가하여 붙이는 순차형으로는 구현이 불가능
# 
# Model()함수로 입력과 출력을 따로 설정하며 모델 만듭니다. 그 다음 compile과 fit은 이전과 동일하게 적용하시면 됩니다. 

#-----------------------------------
#예측 모델 인코더 정의
#-----------------------------------

# 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
# encoder_states = [state_h, state_c]
encoder_model = models.Model(input=encoder_inputs, output=encoder_states) # hidden상태값이 output

#-----------------------------------
# 예측 모델 디코더 정의
#-----------------------------------

# 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
# 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim,),name='B1')
decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim,),name='B2')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 임베딩 레이어
decoder_outputs = decoder_embedding(decoder_inputs) # A4

# LSTM레이어
decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs, initial_state = decoder_states_inputs) # A6

# 히든 상태와 셀 상태를 하나로 묶음
decoder_states = [state_h, state_c]

# Dense레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
decoder_outputs = decoder_dense(decoder_outputs)

# 예측 모델 디코더 설정
decoder_model = models.Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs]+decoder_states)
decoder_model.summary()
 [markdown]
# 예측 모델은 이미 학습된 훈련 모델의 레이어들을 그대로 재사용. 예측 모델 인코더는 훈련 모델 인코더와 동일. 그러나 예측 모델 디코더는 매번 LSTM상태값을 입력 받음. 또한 디코더의 LSTM상태를 출력값과 같이 내보내서, 다음 번 입력에 넣습니다.
# 
# 이렇게 하는 이유는 LSTM을 딱 한번의 타임스텝만 실행하기 때문. 그래서 매번 상태값을 새로 초기화 해야 합니다. 이와 반대로 훈련할 때는 문장 전체를 계속 LSTM으로 돌리기 때문에 자동으로 상태값이 전달됩니다.  [markdown]
# # 훈련 및 테스트

# 인덱스를 문장으롭 변환
def convert_index_to_text(indexs, vocabulary):
    sentence=''

    #모든 문장에 대해서 반복
    for index in indexs:
        if index == END_INDEX:
            #종료 인덱스면 중지
            break
        elif vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            sentence+=vocabulary[OOV_INDEX]
        # 빈칸 추가
        sentence += ' '
    return sentence


# epoch 반복
for epoch in range(20):
    print('Total Epoch:', epoch+1)

    # 훈련 시작
    history = model.fit([x_encoder,x_decoder], # input
                        y_decoder,  # output
                        epochs=100,
                        batch_size=64,
                        verbose=0)
    
    # 정확도와 손실 출력
    print('accuracy:',history.history['acc'][-1])
    print('loss:',history.history['loss'][-1])

    # 문장 예측 테스트
    # (3박 4일 놀러 가고 싶다) -> (여행 은 언제나 좋죠)
    input_encoder = x_encoder[0].reshape(1, x_encoder[2].shape[0]) # (1,30)
    input_decoder = x_decoder[0].reshape(1, x_decoder[2].shape[0])
    results = model.predict([input_encoder, input_decoder])

    # 결과의 원핫인코딩 형식을 인덱스로 변환
    # 1축을 기준으로 가장 높은 값의 위치를 구함
    indexs = np.argmax(results[0],1)

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
    print(sentence)
    print()


x_encoder[2].shape


input_encoder = x_encoder[2].reshape(1,30)
input_encoder.shape


results[0]
print(results[0].shape) #(30,454)
print(results.shape)    #(1,30,454)
 [markdown]
# 학습이 진행될수록 예측 문장이 제대로 생성되는 것을 볼 수 있다. 다만 여기서의 예측은 단순히 테스트를 위한 것이라, 인코더 입력과 디코더 입력 데이터가 동시에 사용. 아래 문장 생성에서는 예측 모델을 적용하기 때문에, 오직 인코더 입력 데이터만 집어 넣습니다. 

# 모델 저장
encoder_model.save(r'D://PROJECT/model/seq2seq_chatbot_encoder_model.h5')
decoder_model.save(r'D://PROJECT/model/seq2seq_chatbot_decoder_model.h5')

# 인덱스 저장
with open(r'D://PROJECT/model/word_to_index.pkl','wb')as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)
with open(r'D://PROJECT/model/index_to_word.pkl','wb')as f:
    pickle.dump(index_to_word,f,pickle.HIGHEST_PROTOCOL)
 [markdown]
# <pickle모듈><br>
# 일반 텍스트를 파일로 저장할 떄는 파일 입출력 이용
# 하지만 리스트나 클래스같은 텍스트가 아닌 자료형은 일반적인 파일 입출력 방법으로는 데이터를 저장하거나 불러올 수 없다. <br>
# pickle모듈을 이용하면 원하는 데이터를 자료형의 변경 없이 파일로 저장하여 그대로 로드할 수 있다. <br>
# pickle로 데이터를 저장하거나 불러올때는 파일을 바이트형식으로 읽거나 써야한다.(wb,rb)
# 
# pickle.load()는 한줄씩 데이터를 읽어오고
# pickle.dump()는 뭉탱이로 읽어옴
#  [markdown]
# # 문장 생성

# 모델 파일 로드
encoder_model = models.load_model(r'D://PROJECT//model/seq2seq_chatbot_encoder_model.h5')
decoder_model = models.load_model(r'D://PROJECT//model/seq2seq_chatbot_decoder_model.h5')

# 인덱스 파일 로드
with open(r'D://PROJECT//model/word_to_index.pkl','rb') as f:
    word_to_index = pickle.load(f)
with open(r'D://PROJECT//model/index_to_word.pkl','rb') as f:
    index_to_word  = pickle.load(f)


# 예측을 위한 입력 생성
def make_predict_input(sentence):

    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences) # 형태소 분석
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT) # 인덱스화

    return input_seq


# 텍스트 생성
def generate_text(input_seq):

    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq) # 인덱스

    # 목표 시퀀스 초기화
    # 문장 1개, 토큰 1개
    target_seq = np.zeros((1,1))

    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0,0] = STA_INDEX

    # 인덱스 초기화
    indexs = []

    # 디코더 타임 스탭 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict([target_seq]+states) # start + encoder_state(문맥벡터)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0,0,:])
        indexs.append(index)

        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break
        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1,1))
        target_seq[0,0] = index     # 다음 출력할 때 input_index

        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c] # 다음 출력할 때 input_state

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)

    return sentence
 [markdown]
# 제일 첫 단어는 START로 시작. 그리고 출력으로 나온 인덱스를 디코더 입력으로 넣고 다시 예측 반복. 상태값을 받아 다시 입력으로 같이 넣는 것에 주의. END태그가 나오면 문장 생성 종료

# 문장을 인덱스로 변환
input_seq = make_predict_input('친구랑 싸웠어요')
input_seq


# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
sentence


# 문장을 인덱스로 변환
input_seq = make_predict_input('친구랑 심하게 싸웠어요')
input_seq


# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
sentence
 [markdown]
# 데이터셋 문장에서는 없던 '같이'를 단어를 추가해 보았습니다. 그래도 비슷한 의미란 것을 파악하여 동일한 답변이 나왔습니다.

# 문장을 인덱스로 변환
input_seq = make_predict_input('친구랑 싸울 것 같아요')
input_seq


sentence = generate_text(input_seq)
sentence


