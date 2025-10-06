# Olist Brazilian E‑Commerce — Customer Retention & Value (Churn Modeling)

**TL;DR** — Use the public **Olist Brazilian E‑Commerce** dataset to quantify churn under an **inactivity rule (6 months)**, engineer customer‑level features (frequency, monetary, sentiment), and ship a **baseline churn model** (Random Forest) that reaches **ROC‑AUC ≈ 0.70**. The repo includes **EDA**, **churn sensitivity**, and **actionable retention playbooks** based on recency and value.

**Tech**: Python (Pandas, NumPy, SciPy, scikit‑learn) • Seaborn/Matplotlib  
**Data**: Olist Brazilian E‑Commerce Public Dataset (orders, customers, items, payments, reviews, products, sellers, categories)  
**Repo map**: [`notebooks/`](notebooks) • [`data/`](data) • [`assets/`](assets)

---

## Table of Contents
- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [How to Reproduce](#how-to-reproduce)
- [Deliverables](#deliverables)
- [Business Recommendations](#business-recommendations)

---

## Background
E‑commerce teams typically lack a clear, repeatable way to **measure churn** and **prioritize retention**. This project turns the Olist dataset into **customer‑level signals** (frequency, monetary, sentiment) and a **baseline classifier** to identify high‑risk accounts. We keep the pipeline simple, reproducible, and business‑ready: define churn cleanly, avoid label leakage, and report model quality with transparent metrics.

---
### Business Question
Which factors best predict **customer churn** and **repeat purchase**, and how can we translate those signals into **retention actions** with the highest ROI?

---

## Dataset
- ## Dataset
- **Source**: Public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (8 CSV files, ~120 MB total).  
- **Core entities** (loaded in the notebook):  
  - `orders`, `customers`, `order_items`, `order_payments`, `order_reviews`, `products`, `sellers`, `product_category_name_translation`.
- **Dates & parsing**: All timestamp columns are parsed to `datetime64` for reliable time filters and cohorting.
- **Customer label (churn)**: A customer is **churned** if their **last purchase date** is **earlier than `max(order_purchase_timestamp) - 6 months`**.  
  - Sensitivity analysis explores 3, 6, 9, and 12‑month cutoffs.

> **Note**: “Recency” (days since last purchase) is great for *analysis* but is excluded from the **training features** to prevent leakage (it’s anchored to the dataset max date).

---

## Methodology

### 1) Data loading & QC
- Read all 8 tables; convert timestamps; check shapes/dtypes/nulls; basic sanity checks.  
- Merge keys where appropriate and build **customer‑level views**.

### 2) Churn definition & sensitivity
- Compute `last_purchase` per customer and apply inactivity cutoff (default **6 months**).
- Report overall **churn rate** and show **sensitivity curve** across 3/6/9/12 months.

### 3) Feature engineering (customer‑level)
- **Frequency**: `order_count` (number of orders).  
- **Monetary**: `payment_value` summed per customer (`monetary`).  
- **Sentiment/Experience**: mean `review_score` per customer (`avg_review`).  
- **Exploratory** only (excluded from model): `recency_days` since last purchase.

### 4) Statistical tests (active vs churned)
- **Mann‑Whitney U** tests for differences in `order_count`, `monetary`, and `avg_review` between active and churned groups.

### 5) Modeling
- **Baseline features**: `["order_count", "monetary", "avg_review"]`.  
- **Train/test split**: 70/30, stratified by label.  
- **Models**:  
  - **Logistic Regression** (`class_weight="balanced"`; `max_iter=1000`) — linear baseline.  
  - **Random Forest** (`n_estimators=200`, `random_state=42`, `class_weight="balanced"`) — non‑linear baseline.  
- **Metrics**: ROC‑AUC (primary), plus classification report on the test set.  
- **Explainability**: Random‑Forest **feature importances** (expect Monetary & Frequency to dominate; Review adds signal).

### 6) Visualization
- Churn by city (with support threshold).  
- Churn sensitivity vs cutoff window.  
- Correlation heatmap for engineered features vs churn.  
- Distributions by label (frequency, spend, reviews).

---

## Key Findings

### From the Notebook (EDA & Modeling)
- **High churn under 6‑month inactivity**: **≈ 70.3%** of customers qualify as churned (using the defined rule).  
- **Recency is the strongest raw indicator**: **correlation ≈ +0.72** with churn — each extra day without a purchase increases risk (used for EDA only).  
- **Monetary matters most**: In the **Random Forest**, spend carries the largest importance; high‑value customers behave differently around churn.  
- **Frequency & sentiment help**: More orders and higher average review scores correlate with retention; Mann‑Whitney tests show churners order **significantly less** (*p* ≈ 0.0000).  
- **Model performance (baseline)**:  
  - **Random Forest**: **ROC‑AUC ≈ 0.70** (preferred starting score for targeting).  
  - **Logistic Regression**: **ROC‑AUC ≈ 0.54** (non‑linear patterns matter).

> These scores are intentionally **baseline** (no heavy tuning, no advanced embeddings). They’re useful for **ranking** customers by risk and launching **quick wins** in retention.

---

## How to Reproduce

1) **Clone** the repo.  

2) **Data Setup**  
   **Source**: Public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (8 CSV files, ~120 MB total).  
   - **Files**:  
     `olist_orders_dataset.csv`, `olist_customers_dataset.csv`, `olist_order_items_dataset.csv`,  
     `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`,  
     `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `product_category_name_translation.csv`.  
   - **Local path (required for notebook)**:  
     Download all CSVs from Kaggle and place them in your local directory, e.g.:  
     ```python
     DATA_DIR = r"D:\Brazilian E-Commerce Public Dataset by Olist" + "\\"
     ```
     Ensure this folder contains all eight CSV files before running the notebook.

3) **Environment Setup**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt


---

## Deliverables
- Notebook: [Brazilian E-Commerce Public Dataset by Olist.ipynb](./Brazilian%20E-Commerce%20Public%20Dataset%20by%20Olist.ipynb)
 
---

## Business Recommendations (Rechecked & Data-Driven)

### 1) Agree on a churn definition that matches your buying cycle

**What we see:** With a **6-month inactivity rule**, the **churn rate is ~70.3%**. Sensitivity shows **3 mo: 89.6%**, **6 mo: 70.3%**, **9 mo: 49.1%**, **12 mo: 29.7%** (your “Churn Sensitivity” plot).  
**Why it matters:** The cutoff dramatically changes the KPI you report and the size of your “save” audience.  
**Action:**  
- If your typical re-purchase cycle is **≤6 months**, keep the **6-month** definition.  
- If you sell slower-moving goods, consider **9–12 months** to avoid labeling normal gaps as “churn”.

---

### 2) Use Recency as your primary operational trigger (not as a training feature)

**What we see:** `recency_days` correlates strongly with churn (**~+0.72**) in EDA; you correctly excluded recency from training to avoid leakage.  
**Why it matters:** Recency is the best early warning signal for operations, even though it’s unsuitable for model training.  
**Action:**  
- Build a **recency ladder** of triggers at **60/90/120/150/180 days** since last purchase. set up automatic actions or campaigns that trigger when a customer has been inactive for a specific number of days — for example, send a reminder at 60 days, a discount at 120 days, and a stronger win-back offer at 180 days.
- The closer a customer is to churning, the more aggressive your re-engagement efforts should be — bigger offers, personal messages, or exclusive deals — since this is your last chance to win them back.

---

### 3) Prioritize high-value customers (Monetary drives the model)

**What we see:** In the Random Forest, **Monetary ≈ 0.99 importance**; **Order Count ≈ 0.00**; **Avg. Review ≈ 0.01**. Directionally, higher spend and more orders associate with retention (Mann-Whitney tests all **p ≈ 0.0000**).  
**Why it matters:** Value segmentation yields more efficient retention spend.  
**Action:**  
- **Tiered playbooks by value** — e.g., top 20% get concierge support, priority shipping, exclusive offers; lower tiers get automated offers — because saving a **high-value** customer is worth more effort and money than saving a **low-value** one.  
- **Track incremental margin** to ensure save costs make sense by tier. This helps you decide whether your retention campaigns are financially worthwhile — especially across value tiers (Platinum, Gold, Silver).

---

### 4) Make reviews a “friction” signal, even if it’s a weaker model feature

**What we see:** The model mostly used Monetary (total spend) to decide who churns —
it didn’t rely much on `Avg. Review`.  
**Why it matters:** It’s still a reliable indicator of experience problems.  
**Action:**  
- When customers leave low review scores (like 1★–2★), you should automatically identify them and take actions (service recovery flows) to fix their experience — before they churn.

---

### 5) Deploy the model for ranking risk; don’t over-index on “accuracy”

**What we see:** Random Forest **ROC-AUC ≈ 0.704** outperforms Logistic (**≈ 0.543**). Reported accuracies (RF ≈ 0.70, LR ≈ 0.49) are not the right success metric for imbalanced churn.  
**Why it matters:** AUC tells you the model is useful for **prioritizing who to target**, not for binary classification.  
**Action:**  
- **Score customers weekly**, target the **top 10–20% risk** for win-back.  
- Choose thresholds based on **cost-benefit** (discount cost vs. expected save value), not raw accuracy.  
- Add **probability calibration** later to support **budget-constrained targeting**. When your probabilities are calibrated, you can:

Sort customers by true churn likelihood (not just relative scores).

Know that if you target those with, say, P(churn) ≥ 0.65,
you’ll reach roughly the number that fits your budget and expected ROI.

This way, your spending is efficient — you save customers where it pays off most.

---

### 6) Run controlled experiments and measure uplift (not clicks)

**What we see:** Clear separation between active vs. churned groups (significant tests) + a moderate AUC means targeting can help, but the **business lift** must be proven.  
**Why it matters:** Only **incremental uplift** validates ROI.  
**Action:**  
- **A/B test** the model-targeted group vs. control on: reactivation in 30/60 days, incremental revenue, and margin. It proves that your churn model works in the real world, not just in the notebook.  
- **Report by value tier** to see where money is best spent. By reporting results by value tier, you discover where your retention budget actually works best. 
- Track **incremental margin**: When you spend money (like discounts, ads, or personal outreach) to “save” a churn-risk customer, check that the extra profit you earn from saving them is more than what you spent to do it.


---


