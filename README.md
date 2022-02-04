# 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회
(주최 : LG AI Research, 주관: 데이콘)

## 개발 환경
"Ubuntu 18.04.5 LTS"

## 라이브러리 버전
python==3.8.2   
albumentations==1.1.0   
catboost==0.26.1   
numpy==1.22.0   
opencv-python==4.5.3.56   
pandas==1.3.5   
scikit-learn==1.0.2   
tensorboard == 2.7.0   
timm==0.4.12   
torch==1.8.1   
tqdm==4.62.3   

## 실행 방법
### 훈련 데이터 전처리와 모델 학습 
- preprocess_for_train.sh 실행 -> 훈련용 전처리 수행
- train.sh 실행 -> catboost와 deep neural network 모델 학습

### 테스트 데이터 전처리와 모델 로드, 추론
- preprocess_for_test.sh 실행 -> 테스트용 전처리 수행
- predict.sh 실행 -> 실제 추론 수행

## 추가 사항
- 구체적인 내용은 각 폴더에 넣었습니다.
- experiment 폴더에는 수행했지만 성능 개선이 뚜렷하지 않았던 내용을 기술하였습니다. 코드도 업로드 하였지만 잘 정리되어 있지는 않습니다.
- competition 폴더에는 실제 사용한 코드들이 있습니다. 모델에 대한 설명 등을 여기에 기술하였습니다. 참고하시면 되겠습니다.
- 위의 bash 파일은 아주 기초적인 수준으로 이루어져 있습니다. 마찬가지로 자세한 정보는 competition 폴더를 참고 바랍니다.
