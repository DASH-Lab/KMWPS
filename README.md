# KMWPS Dataset
Korean-Math-Word-Problem-Solver

Data size : 10,688

# Scripts
## Teacher Model Training
```
# hl : number of head layers
# hs : heas size
# is : FFN (intermediate size)
# dh : number of attention head

python main.py -gpu 0 -hl 12 -hs 768 -is 3072 -dh 12 -exp <output_folder_name>
```

## Distillation to Student Model
```
# hl : number of head layers
# hs : heas size
# is : FFN (intermediate size)
# dh : number of attention head

python main.py --distill -gpu 0 -hl 4 -hs 768 -is 3072 -dh 2 -exp <output_folder_name>
```



## Inference
```
python inference.py --model_pth_name <model_path>

# output file format will be:
# pred : <inference result>
# true : <ground truth equation>
# results : True if pred == true else False
```


#  Question type

<img width="665" alt="1632473753112" src="https://user-images.githubusercontent.com/67318280/134647544-b576a6d8-f041-4213-a41f-71e23022e854.png">


<!-- #![include 1](https://user-images.githubusercontent.com/67318280/134647672-cdd44a4d-c32e-4480-95b3-20c3b2ce0147.png)<br></br> -->
<!-- #![except 1](https://user-images.githubusercontent.com/67318280/134647807-cbbecfd6-e7fd-4393-b56d-43ad88ebfb6b.png) -->


# Quenstion Augmentation Type


### 1. 질문 인식 능력 검증

   #### - 같은 대상 / 다른 질문 구조
  
      윤기가 가진 끈은 178m이고 민영이가 가진 끈은 264m입니다. 두 사람이 가진 끈은 모두 몇 m입니까?

      -> 윤기가 가진 끈은 178m이고 민영이가 가진 끈은 264m입니다. 두 사람이 가진 끈의 차는 몇 m입니까?

   #### - 다른 대상 / 같은 질문 구조
  
        5, 0.5, 12, 4.1 중에서 가장 큰 수와 두 번째로 큰 수의 곱을 구해 보세요.
  
        -> 5, 0.5, 12, 4.1 중에서 가장 작은 수와 두 번째로 작은 수의 합을 구해 보세요.

  #### - 다른 대상 / 다른 질문 구조
  
      1/7짜리 철사를 호석이는 4도막, 정국이는 6도막, 남준이는 5도막 가지고 있습니다. 철사를 가장 많이 가지고 있는 사람은 누구입니까?
  
      -> 1/7짜리 철사를 호석이는 4도막, 정국이는 6도막, 남준이는 5도막 가지고 있습니다. 철사의 총 길이는 얼마인가요?
      

### 2. 추론 능력 검증 

   #### - 관련있는 정보 삽입
  
      호석이는 체리 21개 중에서 5개를 먹었습니다. 남은 체리는 몇 개일까요?
  
      -> 호석이는 체리 21개 중에서 5개를 먹었습니다. 호석이의 동생이 6개를 더 먹었습니다. 남은 체리는 몇 개일까요?

   #### - 관련있는 정보 변경
  
      지민이는 초콜릿을 35개 가지고 있습니다. 그중에서 4개를 동생에게 주었습니다. 지민이에게 남은 초콜릿은 몇 개일까요?
  
      -> 지민이는 초콜릿을 35개 가지고 있습니다. 동생이 초콜릿 4개를 더 주었습니다. 지민이에게 남은 초콜릿은 몇 개일까요?

   #### - 역계산
  
      닭장에 암탉이 20마리, 수탉이 9마리 있습니다. 닭장에 있는 닭은 모두 몇 마리일까요?
  
      -> 닭장에 닭이 29마리 있고 암탉이 20마리 있습니다. 수탉은 모두 몇 마리 일까요?
      

### 3. 구조적 안정성 검증

   #### - 정보 서순 변경
  
      마트에 딸기 맛 요구르트가 50병, 포토 맛 요구르트가 40병 있습니다. 한 상자에 요구르트를 10병씩 담으려고 합니다. 상자는 모두 몇 개가 필요할까요?
  
      -> 마트에 포토 맛 요구르트가 40병, 딸기 맛 요구르트가 50병 있습니다. 한 상자에 요구르트를 10병씩 담으려고 합니다. 상자는 모두 몇 개가 필요할까요?

   #### - 문장 서순 변경
  
      태형이네 학교에는 남학생이 302명, 여학생이 275명 있습니다. 남학생과 여학생 중에서 어느 쪽이 더 많을까요?
  
      -> 태형이네 학교에는 남학생이 302명이 있습늬다. 여학생이 275명 있다고 하면 남학생과 여학생 중에서 어느 쪽이 더 많을까요?
  

   #### - 부차적인 정보 삽입
  
      지민이는 초콜릿을 35개 가지고 있습니다. 그중에서 4개를 동생에게 주었습니다. 지민이에게 남은 초콜릿은 몇 개일까요?
  
      -> 지민이는 초콜릿을 35개 가지고 있습니다. 동생은 22개를 가지고 있습니다. 지민이는 4개를 동생에게 주었습니다. 지민이에게 남은 초콜릿은 몇 개일까요?
     

# Performance
Model|Accuracy (%)|Parameters reduction (%)|Latency reduction(%)
|:---:|:---------:|:--------:|:------:|
KoBERT-base (Teacher)|89.0|-|-|
KoBERT-L6-H12|87.2|29.4|9.1
KoBERT-L4-H12|86.6|39.1|12.4
KoBERT-L1-H12|85.4|53.8|19.0
KoBERT-L6-H12 (Distill)|89.2|29.4|9.2
KoBERT-L4-H12 (Distill)|88.6|39.1|12.4
KoBERT-L1-H12 (Distill)|88.4|53.8|18.9
KoBERT-L4-H8|87.2|39.1|15.7
KoBERT-L4-H4|87.4|39.1|19.9
KoBERT-L4-H2|86.4|39.1|19.3
KoBERT-L4-H8 (Distill)|87.2|39.1|15.6
KoBERT-L4-H4 (Distill)|87.4|39.1|20.0
KoBERT-L4-H2 (Distill)|86.4|39.1|19.2
KoBERT-L1-H12-Hs252-FFN1024 (Distill)|88.6|81.1|24.8


<!-- KoBERT-L6-H12-Hs252-FFN1024 (Distill)|89.2|78.4|
KoBERT-L1-H12-Hs252-FFN1024 (Distill)|88.4|81.1| -->
<!-- KoBERT-L8-H12|-|19.6|1.3944 -->
<!-- 75.2
78.4
79.6
81.1 -->

<!-- [12.46, 11.32, 10.92, 10.09,
10.50, 9.98, 10.06,

9.37
] -->