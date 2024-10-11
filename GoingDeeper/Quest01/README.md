# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 유지희 그루님
- 리뷰어 : 담안용 그루
***지희님의 노트북 파일이 GITHUB에서 열람되지 않아 부득이하게 사진이 없습니다. 추후 보강하겠습니다***

# PRT(Peer Review Template)
- [세모]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - SentencePiece 모델 학습까지 진행되었습니다.
        - ![image](https://github.com/user-attachments/assets/32493f88-f090-4ded-a7bb-f47534b22b19)

    
- [o]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 실험을 8가지 진행한 점이 매우 인상적이었습니다.
    - -1. 배치 사이즈 증가 2. vocab_size(8000->32000)
    - -layer층 다양한 변화
    - 3. dropout, 4.(드롭아웃빼고)GlobalMaxPooling1D) 5. early_stopping
    - 6. 양방향 LSTM제거 7.dense layer추가 8.유닛 수 증가(hidden_units)
         = ![image](https://github.com/user-attachments/assets/61efab5e-b70e-4c97-b02a-7f55b777b7b3)

        
- [o]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 회고 부분에 작성해주셨습니다.
        
- [o]  **4. 회고를 잘 작성했나요?**
    - 넵! 실화과정에서 고민을 잘 볼 수 있었습니다
        - ![image](https://github.com/user-attachments/assets/729f7777-9fa5-47b2-a7f0-4e3ad5cf9033)

        
- [o]  **5. 코드가 간결하고 효율적인가요?**
    - 코드블록으로 함수 같은 효과를 누렸습니다.
    - ![image](https://github.com/user-attachments/assets/258b6ba9-ed7f-4f5a-b6ad-50cf3ae9a8d1)



# 회고(참고 링크 및 코드 개선)
```
# 토크나이저 성능 비교까지 보지 못하였지만, 지희님게서 문제를 해결하고자 빠르게 다양한 시도를 진행하신 점이 인상적이었습니다.
# 주석 넘버링이 보기 좋았습니다. 제가 목차광인인데 오늘의 코드를 보며 잘 정리된 주석도 목차같은 효과를 낼 수 있을 것 같다라는 생각이 들었습니다. 
```
