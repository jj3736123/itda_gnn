# Model2: SBERT-based RGCN Fake Review Detection

## 1. 개요

이 폴더는 YelpZip 데이터셋을 이용한 GNN 기반 fake review 탐지 프로젝트의 두 번째 모델 실험이다.

model1에서는 manual feature 8개와 TF-IDF/SVD 32차원 text feature를 사용했고, 최종 모델은 `RGCN with New User Burst edge`였다. model2에서는 model1의 sampling, split, edge 구조를 그대로 유지한 상태에서 text feature를 SBERT embedding으로 교체하여 성능 개선을 시도하였다.

model2의 핵심 목표는 다음과 같다.

- model1과 동일한 sample, split, edge 조건을 유지한다.
- text feature를 TF-IDF/SVD 32차원에서 SBERT 384차원 embedding으로 교체한다.
- feature 차원 증가로 인한 overfitting을 줄이기 위해 tuned RGCN을 실험한다.
- `NEW_USER_BURST_7D` edge가 model2에서도 유효한지 ablation으로 확인한다.

## 2. 폴더 구조

```text
model2/
├─ notebooks/
│  ├─ 01_setup_from_model1.py
│  ├─ 02_node_features_sbert.py
│  ├─ 03_prepare_data.py
│  ├─ 04_train_rgcn.py
│  ├─ 04_train_rgcn_tuned.py
│  ├─ 05_train_rgcn_without_new_user_burst_tuned.py
│  └─ 06_compare_results.py
│
├─ data/
│  ├─ model2_sampled_reviews.csv
│  ├─ model2_split.csv
│  ├─ model2_edge_list.csv
│  ├─ model2_edge_index.npy
│  ├─ model2_edge_type.npy
│  ├─ model2_node_features.csv
│  ├─ model2_standard_scaler.pkl
│  ├─ model2_tensor_data.pt
│  ├─ model2_pyg_data.pt
│  ├─ model2_rgcn_best.pt
│  ├─ model2_rgcn_tuned_best.pt
│  └─ model2_rgcn_without_new_user_burst_tuned_best.pt
│
├─ configs/
│  ├─ model2_edge_config.json
│  └─ model2_feature_columns.json
│
├─ outputs/
│  ├─ setup_summary.json
│  ├─ node_feature_summary.json
│  ├─ prepare_data_summary.json
│  ├─ rgcn_result.json
│  ├─ rgcn_tuned_result.json
│  ├─ rgcn_without_new_user_burst_tuned_result.json
│  ├─ model2_comparison.csv
│  ├─ model2_comparison.json
│  └─ model2_comparison_markdown.txt
│
└─ report_assets/
   ├─ final_rgcn_all_predictions.csv
   ├─ final_rgcn_tuned_all_predictions.csv
   └─ final_rgcn_without_new_user_burst_tuned_all_predictions.csv
```

## 3. 데이터 조건

model2는 sampling을 새로 하지 않았다. model1에서 사용한 sample, split, edge를 그대로 복사하여 사용하였다.

이렇게 한 이유는 model1과 model2의 비교 조건을 최대한 동일하게 유지하기 위해서이다. sampling이나 split이 바뀌면 성능 차이가 text feature 때문인지, sample 차이 때문인지 해석하기 어려워진다.

model2에서 재사용한 조건은 다음과 같다.

```text
sample node 수: 25,000
product 수: 1,990
user 수: 21,896

label 분포:
normal: 19,868
fake: 5,132
fake 비율: 20.53%

split:
train: 16,000
valid: 4,000
test: 5,000

edge:
RUR: 10,164
PRODUCT_TIME_7D: 274,492
NEW_USER_BURST_7D: 72,210
total directed edge: 356,866
```

## 4. Node Feature

model1에서는 manual feature 8개와 TF-IDF/SVD 32차원을 사용하였다.

model2에서는 manual feature 8개는 거의 유지하되, text feature를 SBERT embedding으로 교체하였다.

### Manual feature

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

model1의 `rating_deviation_from_prod_avg`는 model2에서 `rating_abs_deviation_from_prod_avg`로 교체하였다. 이는 특정 리뷰의 rating이 해당 product의 평균 rating에서 얼마나 벗어나는지를 보기 위한 feature이다.

다음 feature들은 최종적으로 제외하였다.

```text
prod_burst30_ratio
user_fake_prior_like_score
new_low_activity_user
same_user_review_count_in_sample
```

제외 이유는 sampling 기준 또는 edge 설계와 중복될 가능성이 있고, model2의 핵심 비교가 text feature 개선이기 때문이다.

### Text feature

model2에서는 `sentence-transformers/all-MiniLM-L6-v2`를 사용하여 review text를 384차원 SBERT embedding으로 변환하였다.

최종 feature 수는 다음과 같다.

```text
manual feature: 8
SBERT feature: 384
total feature: 392
```

## 5. Edge 설계

model2는 model1의 edge 구조를 그대로 사용하였다.

### RUR

같은 `user_id`가 작성한 review끼리 연결한 edge이다.

```text
directed edge 수: 10,164
```

### PRODUCT_TIME_7D

같은 `prod_id`에 대해 작성일 차이가 7일 이내인 review끼리 연결한 edge이다.

```text
directed edge 수: 274,492
```

### NEW_USER_BURST_7D

같은 `prod_id`에 대해 작성일 차이가 7일 이내이고, 두 review 작성자가 모두 신규 또는 저활동 user인 경우 연결한 edge이다.

신규 또는 저활동 user 조건은 다음과 같다.

```text
user_review_count_global <= 2
user_age_days_at_review <= 30
```

```text
directed edge 수: 72,210
```

이 edge는 특정 product에 신규 또는 저활동 유저들이 짧은 기간 집중되는 fake review pattern을 포착하기 위해 설계하였다.

## 6. 실행 순서

### 1단계: model1 파일 복사

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model2/notebooks/01_setup_from_model1.py
```

model1에서 sample, split, edge 파일을 model2로 복사한다.

### 2단계: SBERT node feature 생성

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model2/notebooks/02_node_features_sbert.py
```

manual feature 8개와 SBERT 384차원 feature를 생성한다.

### 3단계: PyTorch/PyG 학습 데이터 생성

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model2/notebooks/03_prepare_data.py
```

StandardScaler를 train set에만 fit하고, valid/test에는 transform만 적용하였다.

### 4단계: 기본 RGCN 학습

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model2/notebooks/04_train_rgcn.py
```

### 5단계: tuned RGCN 학습

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model2/notebooks/04_train_rgcn_tuned.py
```

### 6단계: New User Burst edge 제거 ablation

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model2/notebooks/05_train_rgcn_without_new_user_burst_tuned.py
```

### 7단계: 결과 비교

```powershell
& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe c:/Users/user/OneDrive/Desktop/ITDA/model2/notebooks/06_compare_results.py
```

## 7. 실험 결과

| Rank | Model | Test PR-AUC | Macro F1 | Accuracy | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|
| 1 | model2_rgcn_tuned_with_new_user_burst | 0.5789 | 0.7075 | 0.8088 | 0.5339 | 0.5370 |
| 2 | model1_rgcn_with_new_user_burst | 0.5689 | 0.7130 | 0.8066 | - | - |
| 3 | model2_rgcn_basic_with_new_user_burst | 0.5661 | 0.7080 | 0.8070 | 0.5287 | 0.5478 |
| 4 | model2_rgcn_tuned_without_new_user_burst | 0.5389 | 0.6948 | 0.8010 | 0.5151 | 0.5146 |

model2의 최종 모델은 `model2_rgcn_tuned_with_new_user_burst`이다.

## 8. 주요 해석

model2의 기본 RGCN은 Test PR-AUC 0.5661로 model1 최종 RGCN의 0.5689보다 낮았다. 즉, SBERT embedding을 단순히 추가하는 것만으로는 성능이 개선되지 않았다.

그러나 feature 차원이 model1의 40개에서 model2의 392개로 크게 증가했기 때문에, overfitting 가능성을 고려하여 tuned RGCN을 추가로 실험하였다. tuned RGCN에서는 hidden dimension을 64에서 32로 줄이고, dropout과 weight decay를 강화하였다.

그 결과 model2 tuned RGCN은 Test PR-AUC 0.5789를 기록하여 model1 최종 모델보다 0.0100 상승하였다.

```text
model1 final RGCN PR-AUC: 0.5689
model2 tuned RGCN PR-AUC: 0.5789
difference: +0.0100
```

따라서 SBERT 기반 text representation은 단순 적용만으로는 충분하지 않았지만, 적절한 regularization과 결합될 때 fake review 탐지 성능 개선에 기여할 수 있었다.

## 9. New User Burst edge ablation

model2에서도 `NEW_USER_BURST_7D` edge의 효과를 확인하기 위해 해당 edge를 제거한 ablation을 수행하였다.

```text
with NEW_USER_BURST_7D:
Test PR-AUC 0.5789

without NEW_USER_BURST_7D:
Test PR-AUC 0.5389

difference:
+0.0400
```

또한 confusion matrix 기준으로도 New User Burst edge를 포함한 모델이 더 나은 결과를 보였다.

```text
with NEW_USER_BURST_7D:
[[3493, 481], [475, 551]]

without NEW_USER_BURST_7D:
[[3477, 497], [498, 528]]
```

New User Burst edge를 포함했을 때 false positive와 false negative가 모두 감소하였다. 이는 신규 또는 저활동 유저들이 같은 product에 짧은 기간 집중되는 관계가 SBERT feature를 사용하는 model2에서도 여전히 중요한 구조적 신호임을 보여준다.

## 10. 최종 결론

model2의 최종 모델은 `SBERT feature + tuned RGCN + NEW_USER_BURST_7D edge` 조합이다.

model2는 model1에 비해 Test PR-AUC를 0.5689에서 0.5789로 개선하였다. 개선 폭이 매우 크지는 않지만, 동일한 sample, split, edge 조건에서 text representation을 강화하고 regularization을 조정했을 때 성능이 향상되었다는 점에서 의미가 있다.

또한 New User Burst edge를 제거하면 PR-AUC가 0.5389로 크게 하락했기 때문에, fake review 탐지에서 text feature뿐 아니라 유저 행동 기반의 구조적 관계도 중요하다는 점을 확인하였다.

따라서 model2는 단순한 text embedding 교체 실험이 아니라, semantic text feature와 fake review의 집단적 작성 패턴을 함께 활용한 RGCN 기반 탐지 모델로 정리할 수 있다.