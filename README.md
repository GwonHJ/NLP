# NLP

## classification.py

졸업프로젝트 : 한국어 문장을 감정 3가지로 분류하기

!!!

부정 : 공포, 슬픔, 혐오

중립 : 중립

긍정 : 놀람, 행복


LSTM이용

데이터셋 : 단발성
max_word : 25000
테스트 데이터셋 : 30%
임베딩 : maxword, 256
lstm : 256
batch_size : 500

best model : epoch2 정확도 0.67001

 -> 과적합, 정확도 낮은 문제
 

데이터셋을 늘렸음


데이터셋 : 단발성 + 연속성
maxword : 25000
테스트 데이터셋 : 30%
임베딩 : maxword, 256
lstm : 256
batch_size : 500
best model2 : epoch2 정확도 0.7285


데이터셋 : 단발성 + 연속성
maxword : 25000
테스트 데이터셋 : 30%
임베딩 : maxword, 128
lstm : 128
batch_size : 500
best model2 : epoch2 정확도 0.7307


해볼것 : 과적합 문제 해결을 위한 드롭아웃 해보기...해보고 안되면...그때 생각해보기...

드롭아웃을 해도 큰 변화 없었음...

데이터셋 : 단발성 + 연속성 + 네이버 쇼핑 리뷰(추가)

긍정에서 놀람을 뻈음



## [Classifying korean sentence](https://github.com/GwonHJ/NLP/tree/master/Classifying%20korean%20sentence) : 한국어 문장 감정 분석


꿈꾸는 아이 대회 https://dacon.io/competitions/official/235664/codeshare/ 출전을 위해서 자연어처리 챗봇 개발 프로젝트

맡은 역할 : KoLPy, Keras, LSTM등을 이용하여 한국어 문장에 대한 감정을 분류

### 자세한 내용은 해당 폴더의 README.md 참고

#### 데이터 출처

  - train 데이터 : https://aihub.or.kr/keti_data_board/language_intelligence 에서 한국어 감정 정보가 포함된 단발성 데이터셋

  - test 데이터 : https://aihub.or.kr/keti_data_board/language_intelligence 에서 한국어 감정 정보가 포함된 연속적 데이터셋 
  
  
 데이터가 7가지 감정으로 분류되어있으나 데이터의 분류가 정확하지 않은 관계로 긍정, 부정 두가지 분류로만 훈련시켰습니다.
 
데이터 전처리 및 분류
https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
를 참고하여 만들었습니다.


------------------------------------------------------------------------------------

## [chatbotUI](https://github.com/GwonHJ/NLP/tree/master/chatboyUI) : 챗봇UI

PyQt를 이용하여 만든 챗봇 UI

Qt Designer를 이용해서 레이아웃을 만들었습니다.(.ui파일)

#### 실행화면

![KakaoTalk_20201112_160735705](https://user-images.githubusercontent.com/45057466/98907315-fa54f980-2501-11eb-9d0a-6c5313c5b3a3.png)

아직 챗봇모델과 연결하지 않은 상태라서 크누아이가 안녕만 답하고 있습니다.

https://wikidocs.net/book/2944
를 참고하여 만들었습니다.

-------------------------------------------------------------------------------------


## [Ex](https://github.com/GwonHJ/NLP/tree/master/Ex) : 위키독스에 있는 '딥 러닝을 이용한 자연어 처리 입문' 실습예제를 따라한 코드

https://wikidocs.net/book/2155
