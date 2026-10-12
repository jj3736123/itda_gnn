# Model 1: RGCN 기반 YelpZip Fake Review Detection

## 1. 프로젝트 개요

본 프로젝트는 YelpZip 리뷰 데이터를 사용하여 GNN 기반 fake review 탐지 모델을 구축한 작업이다.  
리뷰 하나를 하나의 node로 두고, 리뷰 사이의 관계를 edge로 정의하여 개별 리뷰의 속성뿐 아니라 조직적인 리뷰 작성 패턴까지 학습하는 것을 목표로 한다.

모델은 `RGCN with New User Burst`이며, 다음 세 가지 relation을 사용한다.

1. `RUR`
   - 같은 user_id가 작성한 review끼리 연결

2. `PRODUCT_TIME_7D`
   - 같은 prod_id에 대해 7일 이내 작성된 review끼리 연결

3. `NEW_USER_BURST_7D`
   - 같은 prod_id에 대해 7일 이내 작성되었고, 두 review 작성자가 모두 신규/저활동 user인 경우 연결

최종 test 성능은 다음과 같다.

| Model | Test PR-AUC | Test Macro F1 | Test Accuracy |
|---|---:|---:|---:|
| GraphSAGE | 0.5400 | 0.6985 | 0.7988 |
| RGCN without New User Burst | 0.5443 | 0.6976 | 0.7828 |
| RGCN with New User Burst | 0.5689 | 0.7130 | 0.8066 |

따라서 모델은 `RGCN with New User Burst`로 선정하였다.

---

## 2. 데이터

사용 데이터는 YelpZip 원본 리뷰 데이터이다.

원본 주요 컬럼은 다음과 같다.

| Column | Description |
|---|---|
| user_id | 리뷰 작성자 ID |
| prod_id | 식당 또는 product ID |
| rating | 별점 |
| label | 정상/사기 리뷰 label |
| date | 리뷰 작성일 |
| text | 리뷰 본문 |
| tag | 추가 메타 정보 |

원본 label은 다음과 같이 변환하였다.

| Original label | Converted label | Meaning |
|---:|---:|---|
| -1 | 1 | fake review |
| 1 | 0 | normal review |

전처리 후 데이터 크기는 다음과 같다.

| Item | Value |
|---|---:|
| 전체 review 수 | 608,458 |
| normal review 수 | 528,019 |
| fake review 수 | 80,439 |
| fake 비율 | 13.22% |

---

## 3. 폴더 구조

현재 `model1` 폴더 구조는 다음과 같다.

```text
ITDA/
├─ data/
│  └─ yelpzip_preprocessed.csv
│
└─ model1/
   ├─ notebooks/
   │  ├─ 01_sampling.py
   │  ├─ 02_node_features.py
   │  ├─ 03_edges.py
   │  ├─ 04_prepare_data.py
   │  ├─ 05_train_rgcn.py
   │  ├─ 06_train_rgcn_without_new_user_burst.py
   │  ├─ 07_train_graphsage.py
   │  ├─ 08_compare_results.py
   │  ├─ analyze_rgcn_predictions.py
   │  └─ extract_case_studies.py
   │
   ├─ data/
   │  ├─ model1_sampled_reviews.csv
   │  ├─ model1_node_features.csv
   │  ├─ model1_edge_list.csv
   │  ├─ model1_edge_index.npy
   │  ├─ model1_edge_type.npy
   │  ├─ model1_split.csv
   │  ├─ model1_tensor_data.pt
   │  ├─ model1_pyg_data.pt
   │  └─ model1_rgcn_best.pt
   │
   ├─ configs/
   │  ├─ model1_feature_columns.json
   │  └─ model1_edge_config.json
   │
   ├─ outputs/
   │  ├─ sampling_summary.json
   │  ├─ node_feature_summary.json
   │  ├─ edge_summary.json
   │  ├─ prepare_data_summary.json
   │  ├─ rgcn_result.json
   │  ├─ rgcn_without_new_user_burst_result.json
   │  ├─ graphsage_result.json
   │  └─ model_comparison.csv
   │
   └─ report_assets/
      ├─ final_rgcn_all_predictions.csv
      ├─ high_confidence_fake_reviews.csv
      ├─ product_suspicious_summary.csv
      ├─ dashboard_nodes.csv
      ├─ dashboard_edges.csv
      └─ case_studies/