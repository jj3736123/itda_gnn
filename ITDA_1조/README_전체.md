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

Accuracy도 함께 기록하였지만, fake review detection은 class imbalance가 존재하는 문제이기 때문에 Accuracy만으로 모델을 평가하기 어렵다. 대부분의 리뷰를 normal로 예측해도 Accuracy가 높게 나올 수 있기 때문이다.

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

manual feature는 다음과 같다.

```text
rating
rating_extreme
rating_deviation_from_prod_avg
same_rating_prod_7d_count
review_length_log
user_review_count_global
prod_review_count_global
user_age_days_at_review
```

text feature는 review text에 TF-IDF를 적용한 뒤, SVD를 사용하여 32차원으로 축소한 것이다.

### 5.2 Edge Design

Model 1은 3개의 edge relation을 사용하였다.

| Edge | Description |
|---|---|
| RUR | 같은 user_id가 작성한 review끼리 연결 |
| PRODUCT_TIME_7D | 같은 prod_id에 대해 7일 이내 작성된 review끼리 연결 |
| NEW_USER_BURST_7D | 같은 prod_id에 대해 7일 이내 작성되었고, 두 작성자가 모두 신규 또는 저활동 user인 경우 연결 |

신규 또는 저활동 user 조건은 다음과 같다.

```text
user_review_count_global <= 2
user_age_days_at_review <= 30
```

edge 생성 결과는 다음과 같다.

| Edge | Directed Edge Count |
|---|---:|
| RUR | 10,164 |
| PRODUCT_TIME_7D | 274,492 |
| NEW_USER_BURST_7D | 72,210 |
| Total | 356,866 |

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

Model 1의 결과는 RGCN이 GraphSAGE보다 더 적합할 수 있음을 보여준다. GraphSAGE는 edge type을 구분하지 않고 모든 edge를 하나의 이웃 관계로 처리하지만, RGCN은 RUR, PRODUCT_TIME_7D, NEW_USER_BURST_7D relation을 구분하여 학습한다.

또한 `NEW_USER_BURST_7D` edge를 포함한 RGCN이 가장 높은 PR-AUC와 Macro F1을 기록하였다. 이는 신규 또는 저활동 user들이 특정 product에 짧은 기간 집중되는 구조가 fake review 탐지에 유효한 graph signal임을 보여준다.

---

## 6. Model 2: Model 1 + SBERT Text Feature

Model 2는 Model 1을 확장한 모델이다.

Model 2에서는 sample, split, edge structure를 새로 바꾸지 않고 Model 1의 구조를 유지하였다. 대신 text feature를 TF-IDF/SVD에서 SBERT embedding으로 교체하였다.

주요 변화는 다음과 같다.

```text
Model 1:
TF-IDF/SVD 32-dimensional text feature

Model 2:
SBERT 384-dimensional text embedding
```

즉, sparse keyword 기반 text feature에서 semantic text representation으로 확장한 모델이다.

### 6.1 Node Feature

Model 2는 다음 feature를 사용하였다.

```text
manual feature: 8
SBERT feature: 384
total feature: 392
```

manual feature는 대부분 Model 1과 동일하게 유지하였고, `rating_deviation_from_prod_avg`는 `rating_abs_deviation_from_prod_avg`로 교체하였다.

사용한 SBERT model은 다음과 같다.

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 6.2 Edge Design

Model 2는 Model 1과 동일한 edge structure를 사용하였다.

```text
RUR
PRODUCT_TIME_7D
NEW_USER_BURST_7D
```

이렇게 한 이유는 Model 1과 Model 2의 비교 조건을 통제하기 위해서이다. edge와 split을 그대로 유지했기 때문에, 성능 변화는 주로 text representation 변화의 효과로 해석할 수 있다.

### 6.3 Tuning

Model 2에서는 feature dimension이 40개에서 392개로 크게 증가하였다. 이로 인해 overfitting 가능성이 커질 수 있기 때문에 tuned RGCN을 추가로 실험하였다.

tuned RGCN에서는 hidden dimension을 줄이고, dropout과 weight decay를 강화하였다.

```text
hidden dimension 감소
dropout 강화
weight decay 적용
```

### 6.4 Model 2 Result

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

Model 1과 비교하면 다음과 같다.

```text
Model 1 PR-AUC: 0.5689
Model 2 PR-AUC: 0.5789
Improvement: +0.0100
```

Model 2는 SBERT text feature를 적절한 regularization과 함께 사용할 때 fake review 탐지 성능을 개선할 수 있음을 보여준다.

### 6.5 New User Burst Ablation

Model 2에서도 `NEW_USER_BURST_7D` edge의 효과를 확인하기 위해 ablation을 수행하였다.

```text
With NEW_USER_BURST_7D:
Test PR-AUC 0.5789

Without NEW_USER_BURST_7D:
Test PR-AUC 0.5389

Difference:
+0.0400
```

이는 SBERT 기반 text feature를 사용하더라도 fake review detection이 text 정보만으로 충분하지 않다는 것을 보여준다. 신규 또는 저활동 user들이 같은 product에 짧은 기간 집중되는 structural pattern은 여전히 중요한 탐지 신호로 작동하였다.

---

## 7. Model 3: Model 2 + SBERT PCA + TEXT_SIMILARITY Edge

Model 3는 Model 2를 확장한 모델이다.

Model 2에서는 SBERT embedding을 node feature로만 사용하였다. Model 3에서는 SBERT 정보를 두 가지 방식으로 사용하였다.

```text
1. Node feature:
SBERT PCA 128

2. Graph relation:
TEXT_SIMILARITY edge
```

핵심 아이디어는 fake review가 개별 review의 의미뿐 아니라, 서로 다른 user들이 같은 product에 대해 의미적으로 유사한 review를 반복 작성하는 관계에서도 드러날 수 있다는 것이다.

### 7.1 Node Feature

Model 3는 Model 2의 SBERT 384차원 embedding을 PCA를 통해 128차원으로 축소하였다.

```text
manual feature: 8
SBERT PCA feature: 128
total feature: 136
```

PCA는 train split에만 fit하고, valid/test에는 transform만 적용하였다.

```text
PCA source: SBERT 384 feature
PCA components: 128
PCA fit split: train only
explained variance ratio sum: 0.8413
```

이는 semantic information을 최대한 유지하면서 feature dimension을 줄이고, overfitting 가능성을 완화하기 위한 설계이다.

### 7.2 Edge Design

Model 3는 기존 3개 behavioral edge를 유지하고, semantic edge를 추가하였다.

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

prototype TEXT_SIMILARITY edge 생성 결과는 다음과 같다.

```text
TEXT_SIMILARITY directed edge count: 710
base edge count: 356,866
combined edge count: 357,576
number of relations: 4
```

### 7.3 Model 3 Result

| Rank | Model | Test PR-AUC | Macro F1 | Accuracy | Precision | Recall | Threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | model3_rgcn_pca128_text_similarity | 0.5909 | 0.7187 | 0.8120 | 0.5393 | 0.5750 | 0.67 |
| 2 | model3_rgcn_pca128_original_edges | 0.5823 | 0.7070 | 0.7980 | 0.5067 | 0.5867 | 0.64 |
| 3 | model2_rgcn_with_new_user_burst_tuned | 0.5789 | 0.7075 | 0.8088 | 0.5339 | 0.5370 | - |

Model 3의 개선은 두 단계로 나타났다.

첫째, SBERT PCA 128을 적용했을 때 PR-AUC가 소폭 상승하였다.

```text
Model 2 best PR-AUC: 0.5789
Model 3 PCA128 original edge PR-AUC: 0.5823
Improvement: +0.0034
```

둘째, TEXT_SIMILARITY edge를 추가했을 때 PR-AUC가 추가로 상승하였다.

```text
Model 3 PCA128 original edge PR-AUC: 0.5823
Model 3 TEXT_SIMILARITY PR-AUC: 0.5909
Improvement: +0.0086
```

따라서 Model 3는 SBERT가 단순 node feature뿐 아니라 graph relation으로도 활용될 수 있음을 보여준다.

---

## 8. Model3 Final: Model 3 + Validation-based TEXT_SIMILARITY Grid Search

`model3_final`은 이 프로젝트의 최종 선택 모델이다.

이는 기존 Model 3의 이름만 바꾼 것이 아니라, Model 3의 TEXT_SIMILARITY edge 조건을 validation 기준으로 비교하고 보완한 최종 실험이다.

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

manual feature는 다음과 같다.

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

### 8.2 Final Edge Relations

최종 모델은 4개의 edge relation을 사용하였다.

| Relation | Description | Directed Edge Count |
|---|---|---:|
| RUR | 같은 user가 작성한 review끼리 연결 | 10,164 |
| PRODUCT_TIME_7D | 같은 product에 대해 7일 이내 작성된 review끼리 연결 | 274,492 |
| NEW_USER_BURST_7D | 같은 product, 7일 이내, 두 user가 모두 신규 또는 저활동 user인 경우 연결 | 72,210 |
| TEXT_SIMILARITY | 같은 product, 서로 다른 user, 의미적으로 유사한 review text를 연결 | 822 |

최종 TEXT_SIMILARITY 조건은 다음과 같다.

```text
SIM_THRESHOLD = 0.85
TOP_K = 5
MIN_TEXT_LENGTH = 20
```

최종 edge 생성 결과는 다음과 같다.

```text
TEXT_SIMILARITY directed edge count: 822
products with TEXT_SIMILARITY edges: 91
base edge count: 356,866
combined edge count: 357,688
number of relations: 4
```

### 8.3 TEXT_SIMILARITY Grid Search

최종 모델에서는 다음 조합을 비교하였다.

```text
SIM_THRESHOLD: 0.80 / 0.85 / 0.90
TOP_K: 3 / 5
selection rule: highest validation PR-AUC
```

Grid search 결과는 다음과 같다.

| Rank | Candidate | SIM_THRESHOLD | TOP_K | Text Edges | Products with Text Edges | Valid PR-AUC | Test PR-AUC | Macro F1 | Threshold |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | model3_final_thr085_topk5 | 0.85 | 5 | 822 | 91 | 0.5773 | 0.5909 | 0.7212 | 0.68 |
| 2 | model3_final_thr085_topk3 | 0.85 | 3 | 710 | 91 | 0.5755 | 0.5909 | 0.7187 | 0.67 |
| 3 | model3_final_thr09_topk5 | 0.90 | 5 | 580 | 25 | 0.5707 | 0.5864 | 0.7187 | 0.67 |
| 4 | model3_final_thr08_topk3 | 0.80 | 3 | 2,592 | 375 | 0.5699 | 0.5796 | 0.7154 | 0.66 |
| 5 | model3_final_thr09_topk3 | 0.90 | 3 | 476 | 25 | 0.5692 | 0.5883 | 0.7174 | 0.66 |
| 6 | model3_final_thr08_topk5 | 0.80 | 5 | 2,814 | 375 | 0.5690 | 0.5774 | 0.7175 | 0.65 |

validation PR-AUC 기준 최종 선택된 조합은 다음과 같다.

```text
final candidate: model3_final_thr085_topk5
SIM_THRESHOLD: 0.85
TOP_K: 5
MIN_TEXT_LENGTH: 20
valid PR-AUC: 0.5773
test PR-AUC: 0.5909
test Macro F1: 0.7212
```

따라서 `model3_final`은 단순히 기존 Model 3의 이름을 바꾼 것이 아니라, TEXT_SIMILARITY edge 조건을 validation 기준으로 검증한 뒤 선택한 최종 모델이다.

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

case study는 모델 학습에 사용되는 것이 아니라, 최종 모델의 예측 결과를 해석하고 suspicious product를 설명하기 위한 report asset이다.

최종 case study 조건은 다음과 같다.

```text
MIN_REVIEW_COUNT = 20
HIGH_CONF_THRESHOLD = 0.80
```

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