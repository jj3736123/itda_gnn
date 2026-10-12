# Model3: SBERT PCA 128 + TEXT_SIMILARITY Edge

## 1. 개요

model2에서 SBERT 384차원 feature는 tuned RGCN에서만 개선(PR-AUC 0.5789)을 보였고, 차원이 커서 노이즈와 과적합 위험이 남아 있었다.
model3는 이를 두 단계로 나눠 검증한다.

- **Step 1** — edge 구조는 model2와 동일하게 두고, SBERT 384를 PCA 128로 압축한 node feature만 교체
- **Step 2** — Step 1의 feature를 유지한 채, 의미적으로 유사한 리뷰를 직접 연결하는 `TEXT_SIMILARITY` edge 추가
- **Final** — `TEXT_SIMILARITY` 생성 조건(유사도 임계값, top_k)을 validation PR-AUC 기준 grid search로 선택

sample(25,000 reviews, 1,990 products)과 train/valid/test split은 model1·model2와 동일하다.

## 2. Node Feature

| 구성 | 차원 |
| --- | ---: |
| manual feature (rating, rating 극단값, 상품 평균 대비 편차, 7일 내 동일 별점 수 등) | 8 |
| SBERT(all-MiniLM-L6-v2) 384 → PCA 128 | 128 |
| 합계 | 136 |

PCA는 train split에만 fit하고 valid/test에는 transform만 적용해 정보 누수를 막았다. 128차원으로 원본 분산의 약 84%를 보존했다.

## 3. TEXT_SIMILARITY Edge

| 조건 | 값 |
| --- | --- |
| 비교 범위 | 같은 `prod_id` 안에서만 |
| 작성자 | 서로 다른 `user_id` (같은 user는 이미 RUR로 연결) |
| 최소 텍스트 길이 | 20자 ("good food" 같은 짧은 표현의 우연한 유사도 제거) |
| 유사도 | SBERT cosine similarity ≥ threshold |
| 리뷰당 연결 수 | top_k |
| label 사용 | 사용하지 않음 |

prototype(threshold 0.85, top_k 3)에서 710개 directed edge, 91개 상품에서 생성되었다.

## 4. 결과

| 모델 | Test PR-AUC | Macro F1 | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| model2 tuned (기준) | 0.5789 | 0.7075 | 0.8088 | 0.5339 | 0.5370 |
| Step 1: PCA 128 + 기존 edge | 0.5823 | 0.7070 | 0.7980 | 0.5067 | 0.5867 |
| Step 2: PCA 128 + TEXT_SIMILARITY | 0.5909 | 0.7187 | 0.8120 | 0.5393 | 0.5750 |
| **Final (thr 0.85, top_k 5)** | **0.5909** | **0.7212** | **0.8158** | **0.5495** | **0.5682** |

### Grid search

| threshold | top_k | 선택 기준 |
| --- | --- | --- |
| 0.80 / 0.85 / 0.90 | 3 / 5 | validation PR-AUC 최고 → 동률 시 validation Macro F1 |

최종 선택은 threshold 0.85, top_k 5 (TEXT_SIMILARITY 822개 edge). test 성능의 극적 향상이 아니라 validation 기준 가장 안정적인 조건이었고, edge 수가 과도하게 늘지 않으면서 Macro F1이 함께 개선된 점이 선택 근거다.

## 5. 해석

- PCA 압축만으로 +0.0034, 같은 feature에 TEXT_SIMILARITY edge만 추가해 +0.0086 개선되었다.
- 리뷰의 의미 정보는 node feature로만 넣을 때보다 **graph relation으로도 표현할 때** RGCN이 fake 패턴을 더 잘 학습했다.

## 6. 실행 순서

```bash
python model3/notebooks/01_setup_from_model2.py
python model3/notebooks/02_node_features_sbert_pca.py
python model3/notebooks/03_edges_text_similarity.py
python model3/notebooks/04_prepare_data.py
python model3/notebooks/05_train_rgcn_pca_original_edges.py
python model3/notebooks/06_train_rgcn_pca_text_similarity.py
python model3/notebooks/07_compare_results.py
python model3/notebooks/08_case_study.py
python model3/notebooks/09_train_model3_final_grid_search.py
python model3/notebooks/10_case_study_model3_final.py
```

결과 파일은 `outputs/`에 저장되며, 최종 결과는 `outputs/model3_final_result.json`이다.
