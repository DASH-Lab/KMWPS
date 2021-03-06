# KMWPS
Korean-Math-Word-Problem-Solver

![ex_screenshot](./img/distillbert_math.gif)

# Scripts
## Teacher Model Training
```
# hl : number of hidden layers
# hs : hidden size
# is : FFN (intermediate size)
# dh : number of attention head

python main.py -gpu 0 -hl 12 -hs 768 -is 3072 -dh 12 -exp <output_folder_name>
```

## Distillation to Student Model
```
# hl : number of hidden layers
# hs : hidden size
# is : FFN (intermediate size)
# dh : number of attention head

# Change config to run iterative distillation
python main.py --distill -gpu 0 -hl 4 -hs 768 -is 3072 -dh 2 -exp <output_folder_name>
```



## Inference
```
python inference.py --gpu 0 -hl <hidden_layer_size> -hs <hidden_size> -is <FFN_size> -dh <number_of_head> --model_pth_name <model_path>
```

# Web Demo
```
# pip install gradio 
# connect to local server : ex) https://?????.gradio.app

python web_demo.py -hl 4 -hs 252 -is 1024 -dh 12 --model_pth_name <model_path>
```
![ex_screenshot](./img/demo2.gif)
<!-- <iframe id="video" width="320" height="320" src="./img/demo.mp4" frameborder="0">
</iframe> -->

# Application Demo
```
# comming soon!
python application_demo.py -hl 4 -hs 252 -is 1024 -dh 12 --model_pth_name <model_path>
```
![ex_screenshot](./img/application.gif)

# Datasets
Data size : 10,688 문제

##  Question type

<img width="665" alt="1632473753112" src="https://user-images.githubusercontent.com/67318280/134647544-b576a6d8-f041-4213-a41f-71e23022e854.png">


<!-- #![include 1](https://user-images.githubusercontent.com/67318280/134647672-cdd44a4d-c32e-4480-95b3-20c3b2ce0147.png)<br></br> -->
<!-- #![except 1](https://user-images.githubusercontent.com/67318280/134647807-cbbecfd6-e7fd-4393-b56d-43ad88ebfb6b.png) -->


## Quenstion Augmentation Type


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
KoBERT-L6-H12|87.2|29.4|11.5
KoBERT-L4-H12|86.6|39.1|17.0
KoBERT-L1-H12|85.4|53.8|21.3
KoBERT-L6-H12 (Distill)|89.2|29.4|11.3
KoBERT-L4-H12 (Distill)|88.6|39.1|17.3
KoBERT-L1-H12 (Distill)|88.4|53.8|22.1
KoBERT-L8-H12-Hs252-FFN1024|87.8|77.4|16.0
KoBERT-L6-H12-Hs252-FFN1024|87.1|78.4|18.9
KoBERT-L4-H12-Hs252-FFN1024|86.4|79.5|23.0
KoBERT-L1-H12-Hs252-FFN1024|85.1|81.1|27.0
KoBERT-L8-H12-Hs252-FFN1024 (Distill)|89.1|77.4|16.0
KoBERT-L6-H12-Hs252-FFN1024 (Distill)|88.7|78.4|18.8
KoBERT-L4-H12-Hs252-FFN1024 (Distill)|88.2|79.5|23.1
KoBERT-L1-H12-Hs252-FFN1024 (Distill)|87.8|81.1|27.2


