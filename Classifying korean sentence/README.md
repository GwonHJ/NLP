
## 한국어 문장 감정 분석

#### 데이터 출처

  - train 데이터 : https://aihub.or.kr/keti_data_board/language_intelligence 에서 한국어 감정 정보가 포함된 단발성 데이터셋

  - test 데이터 : https://aihub.or.kr/keti_data_board/language_intelligence 에서 한국어 감정 정보가 포함된 연속적 데이터셋 
  
  
 데이터가 7가지 감정으로 분류되어있으나 데이터의 분류가 정확하지 않은 관계로 긍정, 부정 두가지 분류로만 훈련시켰습니다.
 
데이터 전처리 및 분류
https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
참고하여 만들었습니다.

개발 전 해야하는 일 
  1. AIHUB에 가입하여 오픈데이터셋을 받는다.
  2. 데이터셋을 긍정, 부정 두가지로 분류한다 -> 공포, 혐오, 분노, 슬픔 : 0(부정), 행복 : 1(긍정)
  
링크에서는 데이터셋이 txt 파일이기 때문에 그 전 과정이 조금 다름.


- [train.py](https://github.com/GwonHJ/NLP/blob/master/Classifying%20korean%20sentence/train.py) : train데이터셋을 형태소 분석하여 json파일로 만들기
- [test.py](https://github.com/GwonHJ/NLP/blob/master/Classifying%20korean%20sentence/test.py) : test데이터셋을 형태소 분석하여 json파일로 만들기
- [after_json.py](https://github.com/GwonHJ/NLP/blob/master/Classifying%20korean%20sentence/after_json.py) : json파일을 가지고 전처리 및 훈련


<위에 참고한 링크와 다른 부분 설명 : cvs파일을 json파일로 만드는 과정>

```python

data=pd.read_csv('onlybinary_data.csv', sep=",")

#print(data.head()) 출력확인
##print(len(data)) 크기확인
##print(type(data)) 타입확인


train_data = pd.DataFrame(data, columns=['Sentence','Emotion'])

#json에 넣기 위해서 after_data에 담아줌
after_data = []

for index in range(len(train_data)):
    okt = Okt()
    ##실행이 잘 되고 있는지 확인용
    if index%50 ==0:
        print(index)
        
    def tokenize(doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)] 

    tokens= tokenize(train_data['Sentence'][index])    
    feeling=train_data['Emotion'][index]
    feeling=int(feeling)
    ##json파일 형태가 [[단어/형태소, 단어/형태소, ..., 단어/형태],[감정]] 이런 형태가 되어야 하기 때문에 [[tokens, feeling]] 
    after_data = after_data + [[tokens,feeling]]

```
