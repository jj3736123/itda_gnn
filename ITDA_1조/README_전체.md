# YelpZip Fake Review Detection with GNN

## 1. 프로젝트 개요

이 프로젝트는 YelpZip 데이터셋을 이용하여 GNN 기반 fake review 탐지 모델을 구축한 프로젝트이다.

핵심 아이디어는 fake review를 리뷰 하나하나의 독립적인 특성만으로 판단하지 않고, 리뷰 사이의 관계까지 함께 보는 것이다. fake review는 단순히 특정 리뷰의 문장이나 별점에서만 드러나는 것이 아니라, 같은 사용자가 여러 리뷰를 작성하는 패턴, 특정 product에 짧은 기간 동안 리뷰가 몰리는 패턴, 신규 또는 저활동 유저들이 특정 product에 집중되는 패턴, 서로 다른 유저가 의미적으로 유사한 리뷰를 반복해서 작성하는 패턴에서도 나타날 수 있다.

따라서 이 프로젝트에서는 review를 node로 정의하고, user, product, time, text similarity 정보를 바탕으로 edge를 설계하였다.

전체 모델 개발 흐름은 다음과 같다.

```text
Model 1:
manual feature
+ TF-IDF/SVD text feature
+ behavioral graph edge
+ RGCN

Model 2:
Model 1
+ SBERT text embedding
+ tuned RGCN

Model 3:
Model 2
+ SBERT PCA 128
+ TEXT_SIMILARITY edge prototype

Model3 Final:
Model 3
+ validation-based TEXT_SIMILARITY grid search
+ final RGCN model
```

최종 선택된 모델은 `model3_final`이다.

내부 실험명으로는 `model3_final_thr085_topk5`에 해당한다.

---

## 2. 데이터셋

사용 데이터는 YelpZip review dataset이다.

각 row는 하나의 review를 의미한다.

| Column | Description |
|---|---|
| user_id | 리뷰 작성자 ID |
| prod_id | 식당 또는 product ID |
| rating | 리뷰 별점 |
| label | 원본 fake/normal label |
| date | 리뷰 작성일 |
| text | 리뷰 본문 |
| tag | 추가 메타 정보 |

원본 label은 다음과 같이 변환하였다.

| Original label | Converted label | Meaning |
|---:|---:|---|
| -1 | 1 | fake review |
| 1 | 0 | normal review |

전처리 후 전체 데이터는 다음과 같다.

| Item | Value |
|---|---:|
| 전체 review 수 | 608,458 |
| normal review 수 | 528,019 |
| fake review 수 | 80,439 |
| fake 비율 | 13.22% |

---

## 3. Sampling Strategy

모든 모델은 동일한 sampled dataset과 동일한 train/valid/test split을 사용하였다.

이렇게 한 이유는 모델 간 비교 조건을 최대한 동일하게 유지하기 위해서이다. sample이나 split이 바뀌면 성능 차이가 모델 설계 때문인지, 데이터 구성 차이 때문인지 해석하기 어려워진다.

샘플링 방식은 Product-Burst Ratio Sampling이다.

단순 random sampling을 사용하지 않고, 특정 product에 리뷰가 짧은 기간 동안 몰린 구간을 중심으로 subgraph를 구성하였다. fake review는 특정 product를 중심으로 짧은 기간에 집중되는 형태로 나타날 수 있기 때문에, 이러한 burst pattern을 반영한 샘플링 방식을 사용하였다.

샘플링 조건은 다음과 같다.

| Item | Value |
|---|---:|
| 목표 review node 수 | 25,000 |
| product 최소 리뷰 수 | 35 |
| burst window | 30일 |
| 정렬 기준 | burst30_ratio 내림차순 |

`burst30_ratio`는 다음과 같이 계산하였다.

```text
burst30_ratio = max_30d_reviews / prod_review_count
```

즉, 어떤 product의 전체 리뷰 중 가장 리뷰가 많이 몰린 30일 구간이 차지하는 비율이다.

샘플링 조건에서 `product 최소 리뷰 수`를 35개로 둔 이유는 product-level graph structure를 만들기 위해서는 한 product 안에 어느 정도 충분한 review 수가 필요하기 때문이다. 리뷰 수가 너무 적은 product까지 포함하면, 특정 30일 구간에 리뷰가 조금만 몰려도 burst30_ratio가 과도하게 높게 계산될 수 있다. 예를 들어 리뷰가 5개뿐인 product에서 3개가 30일 안에 작성되면 ratio는 높게 나오지만, 이를 조직적인 fake review pattern으로 해석하기에는 표본이 너무 작다. 따라서 최소 리뷰 수 35개를 기준으로 두어, 너무 작은 product에서 생기는 우연한 burst를 줄이고, product 내부에서 의미 있는 review 관계와 edge가 형성될 수 있도록 하였다.

`burst window`를 30일로 둔 이유는 sampling 단계에서는 너무 좁은 기간보다 product 단위의 review 집중 현상을 넓게 포착하는 것이 중요하기 때문이다. fake review가 반드시 하루나 일주일 안에만 몰린다고 보기 어렵고, 일정 기간 동안 나누어 작성될 수도 있다. 따라서 30일 window를 사용하여 특정 product에 리뷰가 비정상적으로 집중되는 후보를 먼저 넓게 찾았다. 이후 실제 edge 설계에서는 `PRODUCT_TIME_7D`, `NEW_USER_BURST_7D`처럼 7일 기준을 사용하여 더 짧은 기간 안의 밀집 관계를 세밀하게 연결하였다.

즉, sampling의 30일 window는 suspicious product 후보를 넓게 잡기 위한 기준이고, edge의 7일 window는 그 안에서 더 강한 review burst 관계를 학습하기 위한 기준이다. 이처럼 sampling과 edge 설계의 기간 기준을 다르게 둔 이유는, sampling 단계에서는 충분한 subgraph를 확보하고, edge 단계에서는 더 직접적인 fake review 행동 패턴을 반영하기 위해서이다.

최종 sampled dataset은 다음과 같다.

| Item | Value |
|---|---:|
| sample node 수 | 25,000 |
| product 수 | 1,990 |
| user 수 | 21,896 |
| normal review 수 | 19,868 |
| fake review 수 | 5,132 |
| fake 비율 | 20.53% |

전체 데이터의 fake 비율은 13.22%였지만, Product-Burst Ratio Sampling 이후 sample의 fake 비율은 20.53%로 증가하였다. 따라서 fake review 탐지 실험에 더 적합한 subgraph를 구성할 수 있었다.

train/valid/test split은 다음과 같다.

| Split | Count | Ratio |
|---|---:|---:|
| train | 16,000 | 64% |
| valid | 4,000 | 16% |
| test | 5,000 | 20% |

label 비율이 유지되도록 stratify를 적용하였다.

---

## 4. 평가 지표

주요 평가 지표는 PR-AUC와 Macro F1이다.

Accuracy도 함께 기록하였지만, fake review detection은 class imbalance가 존재하는 문제이기 때문에 Accuracy만으로 모델을 평가하기 어렵다. 대부분의 리뷰를 normal로 예측해도 Accuracy가 높게 나올 수 있기 때문이다.(약 87%)

따라서 이 프로젝트에서는 다음 기준을 중심으로 모델을 평가하였다.

```text
PR-AUC:
class imbalance 상황에서 fake review 탐지 성능을 평가하기 위한 핵심 지표

Macro F1:
normal review와 fake review 양쪽의 분류 성능을 균형 있게 보기 위한 지표

Accuracy:
보조적으로 확인하는 지표
```

threshold는 test set에서 직접 고르지 않고, validation set 기준으로 선택한 뒤 test set에 적용하였다.

---

## 5. Model 1: Behavioral Edge 기반 Baseline RGCN

Model 1은 이 프로젝트의 첫 번째 GNN 기반 fake review detection 모델이다.

목표는 review-level node feature와 behavior-based edge relation을 이용하여 기본적인 graph-based fake review detection pipeline을 구축하는 것이다.

### 5.1 Node Feature

Model 1은 총 40개의 node feature를 사용하였다.

```text
manual feature: 8
TF-IDF/SVD text feature: 32
total feature: 40
```

manual feature는 review의 별점, 작성자 활동성, product 내 리뷰 분포, 리뷰 길이 등을 반영하기 위해 설계하였다. fake review는 단순히 text 내용에서만 드러나는 것이 아니라, 극단적인 별점, 특정 product 평균과 다른 별점, 짧은 기간에 반복되는 유사한 rating, 신규 또는 저활동 user의 작성 패턴에서도 나타날 수 있기 때문이다.

Model 1에서 사용한 manual feature는 다음과 같다.

| Feature | Meaning | Reason |
|---|---|---|
| rating | 해당 review의 별점 | fake review는 특정 product의 평점을 의도적으로 높이거나 낮추기 위해 특정 rating 패턴을 보일 수 있기 때문에 기본 feature로 사용하였다. |
| rating_extreme | rating이 1점 또는 5점이면 1, 아니면 0 | 극단적인 별점은 fake review에서 자주 의심되는 신호가 될 수 있다. 단순 rating 값과 별도로 극단 여부를 명시적으로 반영하기 위해 사용하였다. |
| rating_deviation_from_prod_avg | 해당 review의 rating이 같은 product의 평균 rating에서 얼마나 벗어나는지 나타내는 값 | 특정 product의 일반적인 평가 흐름과 다르게 지나치게 높거나 낮은 rating을 주는 review를 포착하기 위한 feature이다. |
| same_rating_prod_7d_count | 같은 product에서 7일 이내 같은 rating을 준 review 수 | 짧은 기간 동안 같은 product에 같은 별점이 반복적으로 몰리는 패턴을 포착하기 위한 feature이다. 이는 조직적으로 작성된 review burst를 의심할 수 있는 신호가 된다. |
| review_length_log | review text 길이에 log를 적용한 값 | 리뷰 길이는 text의 기본적인 성격을 나타내는 정보이다. 너무 짧은 review나 비정상적으로 긴 review는 일반 review와 다른 패턴을 보일 수 있기 때문에 사용하였다. log를 적용한 이유는 리뷰 길이 값의 편차가 너무 커지는 것을 완화하기 위해서이다. |
| user_review_count_global | 전체 데이터 기준 해당 user가 작성한 review 수 | user의 활동성을 나타내는 feature이다. review 수가 매우 적은 user는 신규 또는 저활동 user일 가능성이 있고, fake review 작성 패턴과 연결될 수 있다. |
| prod_review_count_global | 전체 데이터 기준 해당 product가 받은 review 수 | product의 전체 review 규모를 나타내는 feature이다. review가 많은 product와 적은 product는 같은 fake probability를 보여도 해석이 달라질 수 있기 때문에 product-level context를 반영하기 위해 사용하였다. |
| user_age_days_at_review | 해당 user의 첫 관측 review 이후 현재 review를 작성하기까지 지난 일수 | user가 얼마나 오래 활동한 계정인지 나타내는 feature이다. 첫 관측 이후 얼마 지나지 않아 특정 product에 review를 작성한 경우 신규 계정성 행동으로 해석될 수 있다. |

text feature는 review text에 TF-IDF를 적용한 뒤, SVD를 사용하여 32차원으로 축소하였다.

```text
text_svd_0 ~ text_svd_31
```

TF-IDF/SVD text feature를 사용한 이유는 Model 1 단계에서 text 정보를 너무 무겁게 사용하지 않으면서도, review 본문에 포함된 단어 기반 정보를 node feature에 반영하기 위해서이다. TF-IDF는 review text에서 중요한 단어 패턴을 수치화하고, SVD는 고차원 text representation을 32차원으로 압축하여 RGCN이 학습 가능한 형태로 만든다.

따라서 Model 1의 node feature는 다음 두 가지 정보를 함께 반영한다.

```text
1. manual feature:
rating, user activity, product context, review length, short-term rating pattern

2. TF-IDF/SVD text feature:
review text의 단어 기반 정보
```

### 5.2 Edge Design

Model 1은 3개의 edge relation을 사용하였다.

| Edge | Description |
|---|---|
| RUR | 같은 user_id가 작성한 review끼리 연결 |
| PRODUCT_TIME_7D | 같은 prod_id에 대해 7일 이내 작성된 review끼리 연결 |
| NEW_USER_BURST_7D | 같은 prod_id에 대해 7일 이내 작성되었고, 두 작성자가 모두 신규 또는 저활동 user인 경우 연결 |

edge는 review 사이의 관계를 표현하기 위해 설계하였다. node feature가 개별 review의 속성을 나타낸다면, edge는 두 review가 어떤 관계로 연결되는지를 나타낸다. fake review는 하나의 review만 보면 잘 드러나지 않을 수 있지만, 여러 review가 같은 user, 같은 product, 짧은 작성 기간, 신규 user 집중 같은 관계 안에서 연결될 때 더 뚜렷하게 나타날 수 있다.

`RUR` edge는 같은 user가 작성한 review끼리 연결하기 위한 relation이다. 같은 user가 여러 review를 작성했다면, 해당 review들은 서로 독립적인 데이터가 아니라 같은 작성자의 행동 패턴을 공유할 수 있다. 따라서 같은 user가 작성한 review들을 연결하여 user-level writing behavior를 GNN이 학습할 수 있도록 하였다.

`PRODUCT_TIME_7D` edge는 같은 product에 대해 7일 이내 작성된 review들을 연결하기 위한 relation이다. fake review는 특정 product를 대상으로 짧은 기간에 집중적으로 작성될 수 있다. 따라서 같은 product에 작성된 review 중에서도 시간적으로 가까운 review들을 연결하여 product-level burst pattern을 반영하였다.

이 정보를 node feature가 아니라 edge로 둔 이유는, `PRODUCT_TIME_7D`가 개별 review 하나의 속성이라기보다 review와 review 사이의 관계이기 때문이다. 예를 들어 어떤 review가 7일 이내에 작성된 review 수를 feature로 가질 수는 있지만, 그것만으로는 그 review가 정확히 어떤 review들과 함께 묶여 있는지 알 수 없다. 반면 edge로 연결하면 GNN이 같은 product 안에서 비슷한 시기에 작성된 review들의 feature를 서로 전달받으며 학습할 수 있다. 즉 단순히 “주변에 몇 개의 review가 있다”가 아니라, “어떤 review들과 연결되어 있는가”를 모델이 직접 활용할 수 있다.

`NEW_USER_BURST_7D` edge는 `PRODUCT_TIME_7D`보다 더 강한 suspicious pattern을 반영하기 위해 설계하였다. 조건은 같은 product, 7일 이내 작성, 그리고 두 review 작성자가 모두 신규 또는 저활동 user인 경우이다.

신규 또는 저활동 user 조건은 다음과 같다.

```text
user_review_count_global <= 2
user_age_days_at_review <= 30
```

이 edge를 추가한 이유는 fake review가 단순히 짧은 기간에 많이 작성되는 것뿐 아니라, 활동 이력이 적은 user들이 특정 product에 집중적으로 등장하는 방식으로 나타날 수 있기 때문이다. 일반적인 product burst는 정상적인 인기 상승이나 이벤트 때문에도 발생할 수 있다. 하지만 그 burst 안에서 신규 또는 저활동 user들이 많이 연결된다면, 이는 더 의심스러운 행동 패턴으로 해석할 수 있다.

`NEW_USER_BURST_7D`를 node feature가 아니라 edge로 둔 이유도 같다. 신규 user 여부 자체는 node feature로 표현할 수 있지만, “신규 user들이 같은 product에 짧은 기간 함께 몰려 있다”는 정보는 개별 node 하나의 속성이 아니라 여러 review 사이의 구조적 관계이다. node feature로만 넣으면 각 review가 신규 user가 쓴 글이라는 사실은 알 수 있지만, 그 신규 user review들이 서로 같은 product와 같은 기간 안에서 연결되어 있는지는 직접 표현하기 어렵다. edge로 설계하면 GNN이 신규 또는 저활동 user review들이 서로 연결된 subgraph 구조를 학습할 수 있다.

따라서 Model 1의 edge 설계는 다음과 같은 역할을 한다.

```text
RUR:
같은 user가 작성한 review들을 연결하여 user-level behavior를 반영한다.

PRODUCT_TIME_7D:
같은 product에 짧은 기간 집중된 review들을 연결하여 product-level burst pattern을 반영한다.

NEW_USER_BURST_7D:
product burst 중에서도 신규 또는 저활동 user가 함께 몰린 더 강한 suspicious pattern을 반영한다.
```

edge 생성 결과는 다음과 같다.

| Edge | Directed Edge Count |
|---|---:|
| RUR | 10,164 |
| PRODUCT_TIME_7D | 274,492 |
| NEW_USER_BURST_7D | 72,210 |
| Total | 356,866 |

`PRODUCT_TIME_7D` edge가 가장 많은 이유는 같은 product 안에서 7일 이내 작성된 review 관계를 폭넓게 연결하기 때문이다. 반면 `NEW_USER_BURST_7D` edge는 그중에서도 두 작성자가 모두 신규 또는 저활동 user인 경우만 연결하므로 edge 수가 더 적다. 이는 의도한 설계와 맞는 결과이다. 즉, `PRODUCT_TIME_7D`는 넓은 product burst 관계를 포착하고, `NEW_USER_BURST_7D`는 그 안에서 더 의심스러운 신규 user burst 관계를 따로 분리하여 학습하도록 만든 relation이다.

### 5.3 Model 1 Result

Model 1에서는 GraphSAGE, RGCN without New User Burst, RGCN with New User Burst를 비교하였다.

| Model | Test PR-AUC | Test Macro F1 | Test Accuracy |
|---|---:|---:|---:|
| GraphSAGE | 0.5400 | 0.6985 | 0.7988 |
| RGCN without New User Burst | 0.5443 | 0.6976 | 0.7828 |
| RGCN with New User Burst | 0.5689 | 0.7130 | 0.8066 |

Model 1의 최종 모델은 다음과 같다.

```text
Model: RGCN with New User Burst
Test PR-AUC: 0.5689
Macro F1: 0.7130
Accuracy: 0.8066
Threshold: 0.68
```

Model 1의 결과는 RGCN이 GraphSAGE보다 더 적합할 수 있음을 보여준다. GraphSAGE는 edge type을 구분하지 않고 모든 edge를 하나의 이웃 관계로 처리하지만, RGCN은 `RUR`, `PRODUCT_TIME_7D`, `NEW_USER_BURST_7D` relation을 구분하여 학습한다. 이 프로젝트에서는 edge마다 의미가 다르기 때문에, relation별로 다른 message passing을 수행할 수 있는 RGCN이 더 적합하다고 판단하였다.

특히 `NEW_USER_BURST_7D` edge의 효과는 ablation 결과에서 확인할 수 있다. 동일한 RGCN 구조에서 `NEW_USER_BURST_7D` edge를 제거한 모델과 포함한 모델을 비교하면 다음과 같다.

```text
RGCN without New User Burst PR-AUC: 0.5443
RGCN with New User Burst PR-AUC: 0.5689
Improvement: +0.0246

RGCN without New User Burst Macro F1: 0.6976
RGCN with New User Burst Macro F1: 0.7130
Improvement: +0.0154

RGCN without New User Burst Accuracy: 0.7828
RGCN with New User Burst Accuracy: 0.8066
Improvement: +0.0238
```

이 비교에서 중요한 점은 모델 구조는 RGCN으로 동일하고, 차이는 `NEW_USER_BURST_7D` relation의 포함 여부라는 것이다. 따라서 성능 향상은 단순히 모델 architecture 차이 때문이 아니라, 신규 또는 저활동 user들이 특정 product에 짧은 기간 집중되는 관계를 edge로 추가한 것이 실제 탐지 성능에 기여했음을 보여준다.

따라서 Model 1의 최종 모델은 `RGCN with New User Burst`로 선택하였다.

---

## 6. Model 2: Model 1 + SBERT Text Feature

Model 2는 Model 1을 확장한 두 번째 실험이다.

Model 1에서는 manual feature 8개와 TF-IDF/SVD 32차원 text feature를 사용하였다. Model 2에서는 sample, split, edge structure는 그대로 유지하고, text feature만 SBERT embedding으로 교체하는 방식으로 실험을 시작하였다.

주요 변화는 다음과 같다.

```text
Model 1:
manual feature 8
+ TF-IDF/SVD text feature 32
+ RGCN

Model 2:
manual feature 8
+ SBERT text embedding 384
+ RGCN
```

sample, split, edge structure를 그대로 유지한 이유는 Model 1과 Model 2의 비교 조건을 최대한 통제하기 위해서이다. 만약 sample이나 edge까지 동시에 바꾸면 성능 차이가 text representation 때문인지, graph structure 때문인지 해석하기 어려워진다.

따라서 Model 2의 첫 번째 목적은 다음 질문을 확인하는 것이었다.

```text
TF-IDF/SVD text feature를 SBERT text embedding으로 바꾸면 성능이 개선되는가?
```

다만 실험 결과, SBERT를 단순히 추가하는 것만으로는 성능이 바로 개선되지 않았다. 따라서 Model 2는 이후 SBERT feature의 높은 차원에 맞게 tuned RGCN을 추가로 실험하였다.

---

### 6.1 왜 SBERT를 사용했는가

Model 1에서 사용한 TF-IDF/SVD text feature는 review text의 단어 기반 정보를 반영할 수 있다는 장점이 있다. 그러나 TF-IDF는 기본적으로 단어의 등장 빈도에 기반하기 때문에, 문장의 전체 의미나 표현 방식이 다른 유사 문장을 충분히 반영하기 어렵다.

예를 들어 두 review가 서로 다른 단어를 사용했더라도 같은 의미의 칭찬이나 불만을 담고 있을 수 있다. 반대로 단어가 일부 비슷하더라도 실제 문장의 의미는 다를 수 있다. TF-IDF/SVD는 이런 sentence-level semantic information을 직접적으로 반영하기 어렵다.

fake review detection에서는 단순히 어떤 단어가 등장했는지만이 아니라, review가 어떤 의미를 전달하는지도 중요하다. 조직적으로 작성된 fake review는 완전히 같은 문장을 반복하지 않더라도, 비슷한 칭찬 표현, 비슷한 불만 표현, 비슷한 경험 서술을 여러 user가 반복하는 방식으로 나타날 수 있다.

따라서 Model 2에서는 review text의 의미적 정보를 더 잘 반영하기 위해 SBERT embedding을 사용하였다.

사용한 SBERT model은 다음과 같다.

```text
sentence-transformers/all-MiniLM-L6-v2
```

이 모델은 각 review text를 384차원 sentence embedding으로 변환한다.

```text
SBERT model:
sentence-transformers/all-MiniLM-L6-v2

embedding dimension:
384
```
384차원을 사용한 이유는, `sentence-transformers/all-MiniLM-L6-v2` 모델이 기본적으로 384차원 sentence embedding을 출력하기 때문이다.
---

### 6.2 Node Feature

Model 2는 총 392개의 node feature를 사용하였다.

```text
manual feature: 8
SBERT feature: 384
total feature: 392
```

manual feature는 한 개를 제외하고 Model 1과 동일하게 유지하였다.

```text
rating
rating_extreme
rating_abs_deviation_from_prod_avg
same_rating_prod_7d_count
review_length_log
user_review_count_global
prod_review_count_global
user_age_days_at_review
```

Model 1의 `rating_deviation_from_prod_avg`는 Model 2에서 `rating_abs_deviation_from_prod_avg`로 교체하였다. 이는 특정 review의 rating이 해당 product의 평균 rating에서 얼마나 벗어나는지를 방향보다 크기 중심으로 보기 위한 것이다.


---

### 6.3 Edge Design

Model 2는 Model 1과 동일한 edge structure를 사용하였다.

```text
RUR
PRODUCT_TIME_7D
NEW_USER_BURST_7D
```

edge structure를 유지한 이유는 Model 2의 실험 목적이 graph structure 변경이 아니라 text representation 변경이었기 때문이다. 즉, Model 2에서는 edge 설계를 새로 바꾸지 않고, 기존 graph structure 위에서 text feature를 TF-IDF/SVD에서 SBERT로 바꾸었을 때 어떤 변화가 나타나는지를 확인하였다.

Model 2에서 사용한 edge count는 Model 1과 동일하다.

| Edge | Directed Edge Count |
|---|---:|
| RUR | 10,164 |
| PRODUCT_TIME_7D | 274,492 |
| NEW_USER_BURST_7D | 72,210 |
| Total | 356,866 |

---

### 6.4 Basic RGCN: SBERT 단독 교체 효과

먼저 Model 2 basic RGCN을 학습하였다.

결과는 다음과 같다.

```text
Model 1 final RGCN PR-AUC: 0.5689
Model 2 basic RGCN PR-AUC: 0.5661
```

즉, SBERT text representation을 단순히 추가한 것만으로는 Model 1보다 성능이 좋아지지 않았다. 오히려 PR-AUC가 0.5689에서 0.5661로 소폭 하락하였다.

따라서 Model 2의 결과를 “SBERT를 넣었기 때문에 바로 성능이 향상되었다”고 해석하면 안 된다. Model 2 basic RGCN 결과만 보면, SBERT text representation의 단독 효과는 확인되지 않았다.

이런 결과가 나온 이유는 feature dimension 증가와 관련이 있다.

```text
Model 1:
40 features

Model 2:
392 features
```

Model 2는 SBERT 384차원 embedding을 추가하면서 review text의 semantic information을 더 풍부하게 사용할 수 있게 되었다. 그러나 feature 수가 크게 늘어나면 모델이 train data에만 존재하는 세부적인 패턴이나 noise까지 과하게 학습할 가능성도 커진다. 즉 overfitting이 될 수 있다.

그래서 다음과 같은 과정을 추가로 진행하였다.
---

### 6.5 Tuned RGCN


tuned RGCN의 목적은 SBERT feature를 사용하되, 모델이 train data를 과하게 외우지 않도록 학습 방식을 조정하는 것이다.

tuned RGCN에서는 다음과 같은 조정을 적용하였다.

| Tuning 요소 | 의미 | 적용 이유 |
|---|---|---|
| hidden dimension 감소 | GNN 내부에서 node를 표현하는 벡터 크기를 줄이는 것 | 모델이 지나치게 복잡해지는 것을 줄이기 위해 사용하였다. |
| dropout 강화 | 학습 중 일부 정보를 임의로 꺼서 학습하는 방법 | 모델이 특정 feature에 과하게 의존하지 않도록 하기 위해 사용하였다. |
| weight decay 적용 | model weight가 너무 커지지 않도록 제한하는 방법 | train data를 과하게 외우는 것을 줄이고 일반화 성능을 높이기 위해 사용하였다. |

결과는 다음과 같다.

```text
Model 2 basic RGCN PR-AUC: 0.5661
Model 2 tuned RGCN PR-AUC: 0.5789
Improvement: +0.0128
```

Model 1과 비교하면 다음과 같다.

```text
Model 1 final RGCN PR-AUC: 0.5689
Model 2 tuned RGCN PR-AUC: 0.5789
Improvement: +0.0100
```

따라서 Model 2의 성능 개선은 순수하게 “SBERT를 넣었기 때문”이 아니라, `SBERT semantic text feature + tuned RGCN` 조합의 효과로 해석하는 것이 적절하다.

---

### 6.6 Model 2 Result

Model 2의 전체 실험 결과는 다음과 같다.

| Rank | Model | Test PR-AUC | Macro F1 | Accuracy | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|
| 1 | model2_rgcn_tuned_with_new_user_burst | 0.5789 | 0.7075 | 0.8088 | 0.5339 | 0.5370 |
| 2 | model1_rgcn_with_new_user_burst | 0.5689 | 0.7130 | 0.8066 | - | - |
| 3 | model2_rgcn_basic_with_new_user_burst | 0.5661 | 0.7080 | 0.8070 | 0.5287 | 0.5478 |
| 4 | model2_rgcn_tuned_without_new_user_burst | 0.5389 | 0.6948 | 0.8010 | 0.5151 | 0.5146 |

Model 2의 최종 모델은 다음과 같다.

```text
Model: model2_rgcn_tuned_with_new_user_burst
Test PR-AUC: 0.5789
Macro F1: 0.7075
Accuracy: 0.8088
```

Model 2에서 중요한 점은 두 가지이다.

```text
1. SBERT를 단순히 추가한 basic RGCN은 Model 1보다 PR-AUC가 낮았다.
2. SBERT feature에 맞춰 tuned RGCN을 적용했을 때 Model 1보다 PR-AUC가 높아졌다.
```

따라서 Model 2는 SBERT text representation이 가능성은 있지만, 높은 차원의 semantic feature를 안정적으로 활용하려면 model tuning이 함께 필요하다는 점을 보여준다.

---

---

## 7. Model 3: SBERT PCA 128 + TEXT_SIMILARITY Edge

Model 3는 Model 2를 확장한 세 번째 실험이다.

Model 2에서는 SBERT 384차원 embedding을 node feature로 사용하였다. 그러나 Model 2 basic RGCN에서는 SBERT를 단순히 추가했을 때 Model 1보다 PR-AUC가 낮았고, tuned RGCN을 적용했을 때 성능이 개선되었다. 이는 SBERT가 담고 있는 semantic information 자체는 유용할 수 있지만, 384차원 feature를 그대로 사용하는 방식은 모델이 안정적으로 활용하기 어려울 수 있음을 보여준다.

따라서 Model 3에서는 SBERT 정보를 더 안정적으로 활용하기 위해 두 단계로 실험을 설계하였다.

```text
Step 1:
SBERT 384차원 embedding을 PCA 128차원으로 축소한다.
기존 edge structure는 그대로 유지한 상태에서 RGCN을 학습한다.

Step 2:
Step 1의 SBERT PCA 128 node feature를 유지한다.
여기에 TEXT_SIMILARITY edge를 추가하여 RGCN을 다시 학습한다.
```

즉, Model 3는 처음부터 PCA와 TEXT_SIMILARITY edge를 한 번에 추가한 것이 아니다. 먼저 SBERT PCA 128 node feature의 효과를 확인하고, 그다음 TEXT_SIMILARITY edge를 추가했을 때의 효과를 따로 확인하였다.

이렇게 실험을 나눈 이유는 성능 개선의 원인을 더 명확하게 보기 위해서이다. 만약 PCA 128과 TEXT_SIMILARITY edge를 동시에 추가하면, 성능이 좋아졌을 때 그것이 feature 차원 축소 때문인지, semantic edge 추가 때문인지 구분하기 어렵다.

따라서 Model 3에서는 다음 두 모델을 비교하였다.

```text
1. model3_rgcn_pca128_original_edges
   manual 8 + SBERT PCA 128 node feature
   + 기존 edge 3개
   + RGCN

2. model3_rgcn_pca128_text_similarity
   manual 8 + SBERT PCA 128 node feature
   + 기존 edge 3개
   + TEXT_SIMILARITY edge
   + RGCN
```


---

### 7.1 Step 1: SBERT PCA 128 Node Feature

Model 3의 첫 번째 변화는 SBERT 384차원 embedding을 그대로 사용하지 않고, PCA를 통해 128차원으로 축소한 것이다.

PCA 설정은 다음과 같다.

```text
PCA source: SBERT 384 feature
PCA components: 128
PCA fit split: train only
explained variance ratio sum: 0.8413
```

PCA를 적용한 결과, 128차원만으로도 원래 SBERT 384차원 feature가 담고 있던 정보 분산의 약 84.13%를 유지할 수 있었다. 쉽게 말하면, feature 수는 384개에서 128개로 줄였지만 SBERT embedding의 주요 semantic information은 상당 부분 보존되었다고 볼 수 있다.

PCA는 train split에만 fit하고, valid/test에는 transform만 적용하였다. PCA는 단순 변환처럼 보이지만, 실제로는 데이터의 분포를 보고 어떤 방향의 정보를 남길지 결정한다. 따라서 전체 데이터에 PCA를 fit하면 valid/test 데이터의 분포 정보가 feature 생성 과정에 미리 반영될 수 있다. 이는 test label을 직접 사용한 것은 아니지만, 평가 데이터의 정보를 학습 과정에 일부 섞는 data leakage가 될 수 있다. 그래서 train 데이터로만 PCA 기준을 정하고, valid/test는 그 기준에 맞춰 변환만 수행하였다.


---

### 7.2 Step 1 Result: PCA 128 + Original Edges

Step 1에서는 Model 2의 edge structure를 그대로 유지하고, node feature만 SBERT 384에서 SBERT PCA 128로 바꿨다.

사용한 edge는 다음 3개이다.

```text
RUR
PRODUCT_TIME_7D
NEW_USER_BURST_7D
```

결과는 다음과 같다.

```text
Model 2 best PR-AUC: 0.5789
Model 3 PCA128 original edge PR-AUC: 0.5823
Improvement: +0.0034
```

이 결과는 SBERT 384차원을 그대로 사용하는 것보다 PCA 128차원으로 압축하는 것이 약간의 성능 개선을 가져왔음을 보여준다. 

---

### 7.3 Step 2: TEXT_SIMILARITY Edge 추가

Model 3의 두 번째 변화는 `TEXT_SIMILARITY` edge를 추가한 것이다.

Model 2와 Model 3 Step 1에서는 SBERT 정보를 node feature로만 사용하였다. 즉, 각 review의 text 의미를 node 자체의 정보로만 반영하였다. 그러나 fake review는 개별 review의 의미뿐 아니라, 서로 다른 user들이 같은 product에 대해 비슷한 의미의 review를 반복적으로 작성하는 관계에서도 나타날 수 있다.

따라서 Model 3 Step 2에서는 SBERT cosine similarity를 이용하여 review 간 semantic similarity를 edge relation으로 추가하였다. 이 edge는 text 의미가 유사한 review들을 직접 연결하여, GNN이 semantic similarity pattern을 graph structure 안에서 학습할 수 있도록 하기 위한 것이다.

Step 2에서 사용한 edge relation은 다음과 같다.

```text
RUR
PRODUCT_TIME_7D
NEW_USER_BURST_7D
TEXT_SIMILARITY
```

`TEXT_SIMILARITY` edge 조건은 다음과 같다.

```text
1. 같은 prod_id 내부에서만 비교
2. 서로 다른 user_id끼리만 연결
3. text length >= 20
4. SBERT cosine similarity >= 0.85
5. 각 review당 top_k = 3
6. 양방향 directed edge로 저장
7. label과 prediction은 edge 생성에 사용하지 않음
```

각 조건을 이렇게 설정한 이유는 다음과 같다.

| Condition | Reason |
|---|---|
| 같은 `prod_id` 내부에서만 비교 | 서로 다른 product의 review는 내용이 비슷해도 fake review 공모 관계로 보기 어렵다. 이 프로젝트의 목적은 특정 product 안에서 반복되는 suspicious review pattern을 찾는 것이므로, text similarity 비교 범위를 같은 product 내부로 제한하였다. |
| 서로 다른 `user_id`끼리만 연결 | 같은 user가 작성한 review는 이미 `RUR` edge로 연결된다. `TEXT_SIMILARITY` edge의 목적은 같은 사람이 쓴 review가 아니라, 서로 다른 user들이 비슷한 의미의 review를 작성한 패턴을 찾는 것이다. 따라서 같은 user의 review끼리는 제외하였다. |
| `text length >= 20` | 너무 짧은 review는 의미적 유사도를 신뢰하기 어렵다. 예를 들어 “good food”, “nice place”처럼 짧고 흔한 표현은 여러 review에서 반복될 수 있지만, 이를 조직적 fake review의 근거로 보기 어렵다. 따라서 일정 길이 이상의 review만 비교하여 의미 없는 similarity edge가 생기는 것을 줄였다. |
| `SBERT cosine similarity >= 0.85` | 유사도가 낮은 review까지 연결하면 edge가 과도하게 많아지고 noise가 증가할 수 있다. Model 3에서는 text similarity edge를 보수적으로 생성하기 위해 0.85 이상의 높은 similarity threshold를 사용하였다. |
| 각 review당 `top_k = 3` | 한 review가 너무 많은 review와 연결되면 특정 product 내부에서 text similarity edge가 과도하게 늘어날 수 있다. 따라서 각 review마다 가장 유사한 상위 3개 review까지만 연결하여 edge 수를 제한하였다. |

여기서 `text length >= 20`, `SIM_THRESHOLD = 0.85`, `top_k = 3`은 외부 논문에서 정해진 표준값이나 데이터에서 최적화된 값이라기보다, Model 3 prototype 단계에서 noise를 줄이기 위해 설정한 보수적 초기값이다.

참고로 `text length >= 20`은 너무 짧고 흔한 review를 제외하기 위한 주관적인 기준이다. 20자가 21자보다 이론적으로 더 낫다는 의미는 아니다. 최소한 짧은 문장 수준의 정보를 가진 review만 비교하기 위해 20자를 기준으로 둔 것이다. `SBERT cosine similarity >= 0.85` 와 `top_k = 3` 도 마찬가지다.


prototype TEXT_SIMILARITY edge 생성 결과는 다음과 같다.

```text
TEXT_SIMILARITY directed edge count: 710
base edge count: 356,866
combined edge count: 357,576
number of relations: 4
```

전체 base edge 수에 비해 `TEXT_SIMILARITY` edge 수는 많지 않다. 이는 Model 3에서 semantic relation을 의도적으로 보수적으로 추가했기 때문이다.

---

### 7.4 Step 2 Result: PCA 128 + TEXT_SIMILARITY Edge

Step 2에서는 Step 1의 SBERT PCA 128 node feature를 그대로 유지하고, 기존 3개 edge에 `TEXT_SIMILARITY` edge를 추가하였다.

결과는 다음과 같다.

```text
Model 3 PCA128 original edge PR-AUC: 0.5823
Model 3 PCA128 TEXT_SIMILARITY PR-AUC: 0.5909
Improvement: +0.0086
```

이는 SBERT 기반 text similarity를 node feature가 아니라 graph relation으로도 반영하는 것이 GNN 학습에 도움이 되었음을 보여준다.

이 단계의 의미는 다음과 같다.

```text
Model 3 Step 1:
manual 8
+ SBERT PCA 128
+ original edge 3개

Model 3 Step 2:
manual 8
+ SBERT PCA 128
+ original edge 3개
+ TEXT_SIMILARITY edge
```

따라서 Step 2의 성능 개선은 PCA 효과가 아니라, 동일한 PCA 128 feature 조건에서 TEXT_SIMILARITY edge를 추가했을 때의 효과로 해석할 수 있다.

---

### 7.5 Model 3 Result

Model 3의 전체 결과는 다음과 같다.

| Rank | Model | Test PR-AUC | Macro F1 | Accuracy | Precision | Recall | Threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | model3_rgcn_pca128_text_similarity | 0.5909 | 0.7187 | 0.8120 | 0.5393 | 0.5750 | 0.67 |
| 2 | model3_rgcn_pca128_original_edges | 0.5823 | 0.7070 | 0.7980 | 0.5067 | 0.5867 | 0.64 |
| 3 | model2_rgcn_with_new_user_burst_tuned | 0.5789 | 0.7075 | 0.8088 | 0.5339 | 0.5370 | - |

Model 3의 개선은 두 단계로 나타났다.

첫째, SBERT PCA 128을 적용했을 때 PR-AUC가 소폭 상승하였다.

둘째, TEXT_SIMILARITY edge를 추가했을 때 PR-AUC가 추가로 상승하였다.


최종적으로 Model 2 best와 Model 3 최종 모델을 비교하면 다음과 같다.

```text
Model 2 best PR-AUC: 0.5789
Model 3 final PR-AUC: 0.5909
Improvement: +0.0120

Model 2 best Macro F1: 0.7075
Model 3 final Macro F1: 0.7187
Improvement: +0.0112
```

따라서 Model 3는 다음 두 가지를 보여준다.

```text
1. SBERT 384차원을 PCA 128로 압축하면,
   semantic information을 유지하면서도 더 안정적인 node feature를 만들 수 있다.

2. SBERT 정보를 node feature로만 사용하는 것보다,
   review 간 semantic similarity를 TEXT_SIMILARITY edge로 추가했을 때
   GNN이 fake review pattern을 더 잘 학습할 수 있다.
```


---

## 8. Model3 Final: Model 3 + Validation-based TEXT_SIMILARITY Grid Search

`model3_final`은 이 프로젝트의 최종 선택 모델이다.

이는  Model 3의 TEXT_SIMILARITY edge 조건을 validation 기준으로 비교하고 보완한 최종 실험이다.

최종 내부 실험명은 다음과 같다.

```text
model3_final_thr085_topk5
```

### 8.1 Final Node Feature

최종 모델은 다음 feature를 사용하였다.

```text
manual feature: 8
SBERT PCA feature: 128
total feature: 136
```

manual feature는 이전과 동일하다.


### 8.2 Final Edge Relations

edge relation은 Model 3와 동일한 4개를 사용하였다.

| Relation | Description | Directed Edge Count |
|---|---|---:|
| RUR | 같은 user가 작성한 review끼리 연결 | 10,164 |
| PRODUCT_TIME_7D | 같은 product에 대해 7일 이내 작성된 review끼리 연결 | 274,492 |
| NEW_USER_BURST_7D | 같은 product, 7일 이내, 두 user가 모두 신규 또는 저활동 user인 경우 연결 | 72,210 |
| TEXT_SIMILARITY | 같은 product, 서로 다른 user, 의미적으로 유사한 review text를 연결 | 822 |

Model3 Final의 TEXT_SIMILARITY 조건은 다음과 같다.

```text
SIM_THRESHOLD = 0.85
TOP_K = 5
MIN_TEXT_LENGTH = 20
```

Model 3 prototype과 비교하면 `SIM_THRESHOLD = 0.85`, `MIN_TEXT_LENGTH = 20`은 그대로 유지되었고, `TOP_K`만 3에서 5로 변경되었다.

`TOP_K`를 5로 변경한 이유는 validation PR-AUC 기준 grid search에서 `SIM_THRESHOLD = 0.85`, `TOP_K = 5` 조합이 가장 좋은 결과를 보였기 때문이다.

Model 3 prototype에서는 `TOP_K = 3`을 사용하여 TEXT_SIMILARITY edge를 매우 보수적으로 추가하였다. 이 방식은 noise를 줄이는 장점이 있지만, 한 review가 연결할 수 있는 semantic neighbor가 최대 3개로 제한되기 때문에 의미적으로 유사한 review 관계를 충분히 반영하지 못할 수 있다.

Model3 Final에서는 이 문제를 확인하기 위해 다음 조합을 비교하였다.

```text
SIM_THRESHOLD: 0.80 / 0.85 / 0.90
TOP_K: 3 / 5
selection rule: highest validation PR-AUC
```

그 결과 `SIM_THRESHOLD = 0.85`, `TOP_K = 5` 조합이 validation PR-AUC 기준으로 가장 높았다.

```text
model3_final_thr085_topk3:
Valid PR-AUC = 0.5755
Test PR-AUC = 0.5909
Macro F1 = 0.7187
TEXT_SIMILARITY edges = 710

model3_final_thr085_topk5:
Valid PR-AUC = 0.5773
Test PR-AUC = 0.5909
Macro F1 = 0.7212
TEXT_SIMILARITY edges = 822
```

즉, `TOP_K = 5`는 `TOP_K = 3`보다 semantic edge를 조금 더 넓게 연결하면서도 성능을 해치지 않았다. 오히려 validation PR-AUC와 Macro F1이 더 좋게 나타났다.
또한 edge 수가 710개에서 822개로 증가했지만, 과도하게 폭증하지는 않았다. 이는 `SIM_THRESHOLD = 0.85`를 유지했기 때문이다. 


---

## 9. Final Model Training Setting

최종 RGCN 학습 설정은 다음과 같다.

```text
model: RGCNBinaryClassifier
hidden_dim: 32
dropout: 0.5
learning_rate: 0.002
weight_decay: 0.0005
epochs: 250
patience: 40
loss: BCEWithLogitsLoss
pos_weight: 3.8706
threshold selection: validation macro F1
model selection: validation PR-AUC
random_state: 42
```

class imbalance를 고려하여 `pos_weight`를 적용하였다.

```text
train negative: 12,715
train positive: 3,285
pos_weight: 3.8706
```

threshold는 validation set에서 선택한 뒤 test set에 적용하였다.

---

## 10. 전체 모델 비교

| Rank | Model | Main Design | Test PR-AUC | Macro F1 | Accuracy |
|---:|---|---|---:|---:|---:|
| 1 | model3_final | Model 3 + validation-selected TEXT_SIMILARITY condition | 0.5909 | 0.7212 | 0.8158 |
| 2 | model3_rgcn_pca128_text_similarity | Model 2 + SBERT PCA 128 + TEXT_SIMILARITY prototype | 0.5909 | 0.7187 | 0.8120 |
| 3 | model3_rgcn_pca128_original_edges | Model 2 + SBERT PCA 128 | 0.5823 | 0.7070 | 0.7980 |
| 4 | model2_rgcn_tuned_with_new_user_burst | Model 1 + SBERT 384 + tuned RGCN | 0.5789 | 0.7075 | 0.8088 |
| 5 | model1_rgcn_with_new_user_burst | Manual + TF-IDF/SVD + behavioral edges | 0.5689 | 0.7130 | 0.8066 |
| 6 | model1_rgcn_without_new_user_burst | Model 1 without NEW_USER_BURST_7D | 0.5443 | 0.6976 | 0.7828 |
| 7 | model1_graphsage | GraphSAGE baseline without relation-specific learning | 0.5400 | 0.6985 | 0.7988 |

Model 1에서 Model3 Final까지의 개선은 다음과 같다.

```text
Model 1 final PR-AUC: 0.5689
Model3 Final PR-AUC: 0.5909
Improvement: +0.0220

Model 1 final Macro F1: 0.7130
Model3 Final Macro F1: 0.7212
Improvement: +0.0082
```

Model 2에서 Model3 Final까지의 개선은 다음과 같다.

```text
Model 2 best PR-AUC: 0.5789
Model3 Final PR-AUC: 0.5909
Improvement: +0.0120

Model 2 best Macro F1: 0.7075
Model3 Final Macro F1: 0.7212
Improvement: +0.0137
```

---

## 11. Final Model Result

최종 모델은 다음과 같다.

```text
model3_final

internal experiment name:
model3_final_thr085_topk5
```

최종 모델 구조는 다음과 같다.

```text
manual 8 features
+ SBERT PCA 128 features
+ RUR edge
+ PRODUCT_TIME_7D edge
+ NEW_USER_BURST_7D edge
+ TEXT_SIMILARITY edge
+ RGCN
```

최종 test 결과는 다음과 같다.

| Metric | Value |
|---|---:|
| Test PR-AUC | 0.5909 |
| Macro F1 | 0.7212 |
| Accuracy | 0.8158 |
| Precision | 0.5495 |
| Recall | 0.5682 |
| Threshold | 0.68 |

Confusion matrix는 다음과 같다.

|  | Pred Normal | Pred Fake |
|---|---:|---:|
| True Normal | 3496 | 478 |
| True Fake | 443 | 583 |

---

## 12. Case Study and Report Assets

case study 파일은 여러 모델 단계에서 생성되었지만, 최종 README에서는 `model3_final`의 case study 결과를 중심으로 정리한다.

case study는 모델 학습에 사용되는 것이 아니라, 최종 모델의 예측 결과를 해석하고 suspicious product를 설명하기 위한 report asset이다. 즉, 모델 성능을 높이기 위한 입력 데이터가 아니라, 최종 모델이 어떤 product를 왜 의심스럽게 판단했는지 설명하기 위한 분석 결과이다.

최종 case study 조건은 다음과 같다.

```text
MIN_REVIEW_COUNT = 20
HIGH_CONF_THRESHOLD = 0.80
```

`MIN_REVIEW_COUNT = 20`을 사용한 이유는 review 수가 너무 적은 product를 대표 사례에서 제외하기 위해서이다. review 수가 5개나 10개 정도로 너무 적으면 평균 fake probability나 pred fake ratio가 높게 나올 수는 있지만, 이를 product-level suspicious pattern으로 해석하기에는 표본이 부족하다. 따라서 최소 20개 이상의 review가 있는 product만 case study 후보로 사용하였다.

`HIGH_CONF_THRESHOLD = 0.80`은 모델이 강하게 fake로 의심한 review를 따로 보기 위한 기준이다. 단순히 threshold를 넘겨 fake로 분류된 review만 보는 것이 아니라, fake probability가 0.80 이상인 high-confidence fake review 비율을 함께 확인하여 모델이 얼마나 강하게 해당 product를 의심했는지 반영하였다.

대표 suspicious product는 다음 요소를 함께 고려하여 선정하였다.

```text
avg_fake_probability
pred_fake_ratio
high_conf_fake_ratio
NEW_USER_BURST_7D edge density
TEXT_SIMILARITY edge density
avg_text_similarity
review_count
```

각 요소를 고려한 이유는 다음과 같다.

| Factor | Meaning | Reason |
|---|---|---|
| avg_fake_probability | 해당 product에 속한 review들의 평균 fake probability | product 전체가 모델에게 얼마나 의심스럽게 평가되었는지를 보기 위한 기준이다. 평균 fake probability가 높으면 특정 review 몇 개만이 아니라 product 전반에서 fake 가능성이 높게 나타났다는 뜻이다. |
| pred_fake_ratio | 해당 product의 review 중 fake로 예측된 review 비율 | product 내부에서 fake로 분류된 review가 얼마나 많이 분포하는지 보기 위한 기준이다. 평균 확률이 높더라도 fake 예측 비율이 낮으면 대표적인 suspicious product로 보기 어렵기 때문에 함께 사용하였다. |
| high_conf_fake_ratio | fake probability가 0.80 이상인 review 비율 | 모델이 강하게 fake라고 판단한 review가 얼마나 많은지 확인하기 위한 기준이다. 단순 fake 예측보다 더 강한 의심 신호를 보기 위해 사용하였다. |
| NEW_USER_BURST_7D edge density | product 내부에서 신규 또는 저활동 user burst edge가 얼마나 조밀한지 나타내는 값 | Model 1부터 중요하게 확인된 behavioral suspicious pattern을 반영하기 위한 기준이다. 신규 또는 저활동 user들이 짧은 기간에 몰려 있으면 조직적 fake review 가능성을 설명하기 좋다. |
| TEXT_SIMILARITY edge density | product 내부에서 text similarity edge가 얼마나 조밀한지 나타내는 값 | Model 3와 Model3 Final의 핵심 개선점인 semantic similarity pattern을 반영하기 위한 기준이다. 서로 다른 user들이 의미적으로 유사한 review를 반복 작성했는지 확인하기 위해 사용하였다. |
| avg_text_similarity | product 내부 text similarity edge들의 평균 유사도 | 단순히 text similarity edge 수만 보는 것이 아니라, 연결된 review들이 실제로 얼마나 강하게 유사한지도 확인하기 위한 기준이다. |
| review_count | 해당 product의 review 수 | case study의 해석 안정성을 위해 사용하였다. review 수가 너무 적으면 높은 fake probability나 edge density가 우연히 나타날 수 있기 때문에, product 규모를 함께 고려하였다. |

이 요소들을 함께 사용한 이유는 suspicious product를 단순히 fake probability 하나로만 선정하지 않기 위해서이다. fake review 의심 사례는 모델의 예측 점수뿐 아니라, product 내부의 behavioral burst pattern과 semantic similarity pattern이 함께 나타날 때 더 설득력 있게 설명할 수 있다.

따라서 최종 case study는 다음 세 가지 관점을 함께 반영한다.

```text
1. Prediction signal:
avg_fake_probability, pred_fake_ratio, high_conf_fake_ratio

2. Behavioral graph signal:
NEW_USER_BURST_7D edge density

3. Semantic graph signal:
TEXT_SIMILARITY edge density, avg_text_similarity
```

review_count가 20개 이상인 product는 총 290개였다.

### 12.1 Top Suspicious Products

| Rank | prod_id | Reviews | New User Burst Edges | Text Similarity Edges | Representative Score |
|---:|---:|---:|---:|---:|---:|
| 1 | 4558 | 24 | 382 | 58 | 0.9208 |
| 2 | 2548 | 23 | 308 | 0 | 0.7350 |
| 3 | 3962 | 56 | 1,948 | 0 | 0.6607 |
| 4 | 247 | 52 | 822 | 4 | 0.6559 |
| 5 | 2397 | 34 | 428 | 0 | 0.6456 |
| 6 | 56 | 188 | 3,748 | 258 | 0.6362 |
| 7 | 4625 | 25 | 184 | 2 | 0.6323 |
| 8 | 3074 | 22 | 274 | 0 | 0.6201 |
| 9 | 2879 | 141 | 8,836 | 0 | 0.5804 |
| 10 | 3392 | 27 | 298 | 0 | 0.5776 |

### 12.2 Case Study Interpretation

`prod_id 4558`은 behavioral pattern과 semantic pattern이 함께 나타나는 대표 사례이다.

이 product는 review_count가 24개로 아주 크지는 않지만, 내부에서 `NEW_USER_BURST_7D` edge가 382개, `TEXT_SIMILARITY` edge가 58개 생성되었다. 이는 신규 또는 저활동 user들이 짧은 기간에 특정 product에 집중되었고, 동시에 서로 다른 user 사이에서 의미적으로 유사한 review 관계도 나타났다는 뜻이다.

`prod_id 56`은 review_count가 188개로 표본 규모가 크고, `TEXT_SIMILARITY` edge가 258개 생성되었다. 이는 semantic similarity pattern이 단순히 리뷰 수가 적은 작은 product에서만 우연히 나타난 것이 아니라, 비교적 큰 product-level review network에서도 관찰될 수 있음을 보여준다.

최종 case study가 주장하는 핵심은 모든 suspicious product에서 text similarity가 강하게 나타난다는 것이 아니다. fake review 의심 신호는 다음 두 가지 방식으로 나타날 수 있다.

```text
1. Behavioral burst pattern:
신규 또는 저활동 user들이 같은 product에 짧은 기간 동안 review를 집중적으로 작성하는 패턴

2. Semantic similarity pattern:
서로 다른 user들이 같은 product에 대해 의미적으로 유사한 review를 작성하는 패턴
```

최종 모델은 이 두 가지 pattern을 graph structure 안에서 함께 반영한다.

---

## 13. 주요 산출물

최종 모델에서 생성된 주요 파일은 다음과 같다.

```text
model3/data/
├─ model3_final_best.pt
├─ model3_final_edge_index.npy
├─ model3_final_edge_type.npy
└─ model3_final_text_similarity_edge_list.csv

model3/configs/
└─ model3_final_edge_config.json

model3/outputs/
├─ model3_final_result.json
├─ model3_final_grid_search_summary.csv
├─ model3_final_grid_search_summary.json
└─ model3_final_case_study_summary.json

model3/report_assets/
├─ model3_final_all_predictions.csv
└─ model3_final_case_studies/
   ├─ model3_final_product_suspicious_summary.csv
   ├─ model3_final_high_confidence_fake_reviews.csv
   ├─ model3_final_dashboard_nodes.csv
   ├─ model3_final_dashboard_edges.csv
   └─ model3_final_representative_case_study_summary.md
```

---

## 14. Repository Structure

프로젝트는 모델 단계별로 구성되어 있다.

```text
ITDA/
├─ data/
│  └─ yelpzip_preprocessed.csv
│
├─ model1/
│  ├─ notebooks/
│  ├─ data/
│  ├─ configs/
│  ├─ outputs/
│  └─ report_assets/
│
├─ model2/
│  ├─ notebooks/
│  ├─ data/
│  ├─ configs/
│  ├─ outputs/
│  └─ report_assets/
│
└─ model3/
   ├─ notebooks/
   ├─ data/
   ├─ configs/
   ├─ outputs/
   └─ report_assets/
```

각 model folder의 역할은 다음과 같다.

```text
model1:
behavioral edge 기반 기본 RGCN fake review detection pipeline

model2:
model1 structure + SBERT node feature + tuned RGCN

model3:
model2 structure + SBERT PCA 128 + TEXT_SIMILARITY prototype

model3_final:
validation-based TEXT_SIMILARITY grid search를 적용한 최종 선택 모델
```

---

## 15. 실행 순서

작업은 주로 Windows VS Code 환경에서 수행하였다.

`py` 명령이 제대로 작동하지 않을 경우 Python 실행 파일을 직접 지정할 수 있다.

예시는 다음과 같다.

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model3/notebooks/10_case_study_model3_final.py
```

### 15.1 Model 1

```powershell
py model1/notebooks/01_sampling.py
py model1/notebooks/02_node_features.py
py model1/notebooks/03_edges.py
py model1/notebooks/04_prepare_data.py
py model1/notebooks/05_train_rgcn.py
py model1/notebooks/06_train_rgcn_without_new_user_burst.py
py model1/notebooks/07_train_graphsage.py
py model1/notebooks/08_compare_results.py
py model1/notebooks/analyze_rgcn_predictions.py
py model1/notebooks/extract_case_studies.py
```

### 15.2 Model 2

```powershell
py model2/notebooks/01_setup_from_model1.py
py model2/notebooks/02_node_features_sbert.py
py model2/notebooks/03_prepare_data.py
py model2/notebooks/04_train_rgcn.py
py model2/notebooks/04_train_rgcn_tuned.py
py model2/notebooks/05_train_rgcn_without_new_user_burst_tuned.py
py model2/notebooks/06_compare_results.py
```

### 15.3 Model 3

```powershell
py model3/notebooks/01_setup_from_model2.py
py model3/notebooks/02_node_features_sbert_pca.py
py model3/notebooks/03_edges_text_similarity.py
py model3/notebooks/04_prepare_data.py
py model3/notebooks/05_train_rgcn_pca_original_edges.py
py model3/notebooks/06_train_rgcn_pca_text_similarity.py
py model3/notebooks/07_compare_results.py
py model3/notebooks/08_case_study.py
```

### 15.4 Model3 Final

```powershell
py model3/notebooks/09_model3_final_grid_search.py
py model3/notebooks/10_case_study_model3_final.py
```

---

## 16. 최종 모델 선택 이유

최종 선택 모델은 `model3_final`이다.

이 모델은 가장 높은 최종 성능을 기록했을 뿐 아니라, Model 1, Model 2, Model 3와의 비교 조건도 유지하였다. 동일한 sample과 split을 사용했고, RGCN architecture도 유지했기 때문에 성능 개선이 feature와 edge 설계의 변화에서 비롯되었음을 비교적 명확하게 해석할 수 있다.

최종 모델의 개선은 다음 세 가지 설계에서 나왔다.

```text
1. SBERT text information을 PCA 128 node feature로 압축하였다.
2. review 간 semantic similarity를 TEXT_SIMILARITY graph relation으로 추가하였다.
3. TEXT_SIMILARITY edge 조건을 validation PR-AUC 기준으로 선택하였다.
```

최종 모델은 다음 성능을 기록하였다.

```text
Test PR-AUC: 0.5909
Macro F1: 0.7212
Accuracy: 0.8158
```

따라서 최종 모델은 fake review detection에서 다음 요소들을 함께 사용하는 것이 효과적임을 보여준다.

```text
review-level feature
+ user/product/time-based behavioral edge
+ text semantic similarity edge
+ relation-aware RGCN learning
```

최종 모델은 다음과 같이 정리할 수 있다.

```text
model3_final

manual 8 features
+ SBERT PCA 128 features
+ RUR
+ PRODUCT_TIME_7D
+ NEW_USER_BURST_7D
+ TEXT_SIMILARITY
+ validation-selected threshold/top_k
+ RGCN
```