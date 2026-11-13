# Benchmark Execution Report: `textoir_banking77`

- **Active Cloud Provider:** `openai` (via `llm-gateway-sdk`)
- **Embedding Model:** `text-embedding-3-small` (1536 dims)
- **Chat / Taxonomy Model:** `gpt-4o-mini`
- **Total Runtime:** `94.2s`

---

## 1. Schema Audit, Missing Value Handling & Leakage-Free Split
- **Total Rows Processed:** `397`
- **Missing Text Rows Retained (with `is_text_missing` indicator):** `0`
- **Split Distribution (Strict Fit-on-Train-Only):** `{'train': 293, 'test': 72, 'val': 32}`
- **Ground-Truth Categories Present:** `8`

## 2. Stage-1/2 Deduplication & SQLite SHA-256 Embedding Cache
- **Input Tickets:** `397` $\rightarrow$ **Unique Contextual Texts (Stage-1 Exact):** `397`
- **Unique Structural Templates (Stage-2 Funnel):** `397`
- **SQLite Cache Hits:** `233` | **New Cloud Gateway Embeddings:** `164` (`3` batch calls)

## 3. Unsupervised Clustering, Silhouette Sweep & Hungarian Alignment
- **Optimal Cluster Count ($K$):** `12` (DeepAligned dropout viable $K$ = `11`, boundary limit reached = `True`)
- **Best Cosine Silhouette Score:** `0.4523`
- **Silhouette Score by Candidate $K$:** `{'5': 0.3897, '6': 0.3995, '7': 0.4201, '8': 0.4274, '9': 0.4329, '10': 0.4276, '11': 0.4377, '12': 0.4523}`
- **External Validity vs Ground-Truth Labels:** `{'test_NMI_vs_ground_truth': 0.8337, 'test_ARI_vs_ground_truth': 0.7178, 'test_Hungarian_ACC_vs_ground_truth': 0.7778, 'full_NMI_vs_ground_truth': 0.8225, 'full_ARI_vs_ground_truth': 0.742, 'full_Hungarian_ACC_vs_ground_truth': 0.7683}`

### Discovered Cluster Taxonomy (c-TF-IDF Keywords + Cloud LLM Naming)
| Cluster ID | Train Size | Short Label | Routing Queue | Top c-TF-IDF Keywords | Signature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `C0` | 19 | **ATM Cash Discrepancy** | `ATM Claims Support` | `atm, amount, pound, gave, atm gave` | Customers report receiving incorrect cash amounts from ATMs, seeking resolution for discrepancies. |
| `C1` | 28 | **Transfer Fees Inquiry** | `Billing Support` | `transfer, fee, fee transfer, charge, transfer fee` | Customers are inquiring about unexpected fees associated with money transfers. |
| `C2` | 28 | **Unknown Direct Debits** | `Fraud Investigation` | `direct, direct debit, debit, debit payment, payment` | Customers report unauthorized direct debit payments not initiated by them. |
| `C3` | 29 | **Card Charge Fees** | `Billing Support` | `card, charge, fee, with card, card charge` | Customers are inquiring about unexpected fees associated with card payments. |
| `C4` | 18 | **Payment Disputes** | `Payment Disputes Team` | `couple, payment, charge, card, back` | Customers are reporting double charges and seeking to dispute unauthorized transactions for refunds. |
| `C5` | 25 | **Cash Transfer Issues** | `Cash Management Support` | `amount, receive, transfer, money, cash` | Customers report discrepancies in cash amounts received or withdrawn. |
| `C6` | 33 | **ATM Cash Withdrawal Issues** | `ATM Support` | `atm, cash, try, cash atm, get cash` | Customers are experiencing problems with cash withdrawals from ATMs, seeking assistance. |
| `C7` | 10 | **Duplicate Transactions** | `Transaction Support` | `transaction, transaction show, show, multiple, time` | Users report transactions appearing multiple times in their accounts, causing confusion. |
| `C8` | 37 | **Deposit Issues** | `Deposit Support` | `deposit, cheque, cash deposit, cash, account` | Customers report discrepancies with deposits not reflecting in their account balances. |
| `C9` | 18 | **Double Charge Issues** | `Billing Support` | `charge, charge twice, twice, double, double charge` | Customers report being charged twice for a single transaction, seeking clarification and resolution. |
| `C10` | 33 | **Withdrawal Fees Inquiry** | `Billing Support` | `charge, fee, withdrawal, cash, withdraw` | Customers are inquiring about unexpected fees charged during cash withdrawals. |
| `C11` | 15 | **Payment Fees Inquiry** | `Billing Support` | `fees, fee, payment, charge, customer` | Customers are inquiring about unexpected fees related to their payments and account management. |

## 4. Downstream Classifier Comparison (with 95% Bootstrap CIs)
- **Selected Production Model:** `logistic_regression`
- **Calibrated Open-Set Cosine Distance Threshold:** `0.3502`

### True End-to-End Ground-Truth Test Performance
| Model | Val Macro-F1 | Test Ground-Truth Accuracy (95% CI) | Test Ground-Truth Macro-F1 (95% CI) | Latency (ms/ticket) |
| :--- | :--- | :--- | :--- | :--- |
| `naive_majority` | 0.0202 | **0.125** (`[0.0556, 0.2083]`) | **0.0278** (`[0.0132, 0.0431]`) | 0.0117 ms |
| `logistic_regression` | 0.8931 | **0.875** (`[0.8056, 0.9444]`) | **0.8753** (`[0.7956, 0.9423]`) | 0.0161 ms |
| `calibrated_linear_svc` | 0.8931 | **0.8611** (`[0.7778, 0.9306]`) | **0.8617** (`[0.7758, 0.914]`) | 0.0764 ms |
| `gradient_boosting` | 0.6278 | **0.7083** (`[0.5969, 0.8198]`) | **0.7026** (`[0.5911, 0.8059]`) | 0.0331 ms |

### Cluster Agreement Metrics (Prediction vs Discovered Cluster)
| Model | Val Macro-F1 | Cluster Agreement Acc (95% CI) | Cluster Macro-F1 (95% CI) | Weighted-F1 | Latency (ms/ticket) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `naive_majority` | 0.0202 | **0.1389** (`[0.0694, 0.2222]`) | **0.0222** (`[0.0118, 0.0331]`) | 0.0339 | 0.0117 ms |
| `logistic_regression` | 0.8931 | **0.9722** (`[0.9441, 1.0]`) | **0.9624** (`[0.8803, 1.0]`) | 0.9746 | 0.0161 ms |
| `calibrated_linear_svc` | 0.8931 | **0.9583** (`[0.9028, 1.0]`) | **0.9495** (`[0.8722, 1.0]`) | 0.963 | 0.0764 ms |
| `gradient_boosting` | 0.6278 | **0.75** (`[0.6382, 0.8476]`) | **0.6808** (`[0.5707, 0.7614]`) | 0.7612 | 0.0331 ms |

## 5. Slice-Based Error Analysis & Quantitative Open-Set Evaluation
- **Short Tickets ($\le$ median length):** Count = `36`, Macro-F1 = `1.0`, Accuracy = `1.0`
- **Long Tickets ($>$ median length):** Count = `36`, Macro-F1 = `0.8639`, Accuracy = `0.9444`
- **Open-Set Novel/Out-of-Distribution Flagged on Test:** `3` (In-Distribution FPR: `0.0417`)
- **Open-Set Out-of-Scope (OOS) Probe Recall:** `1.0` (tested on `6` out-of-scope probe tickets)

## 6. Generated Visual Artifacts
- 2D Cluster & Ground-Truth Scatterplot: `data/artifacts/clusters_2d_projection.png`
- Test Confusion Matrix Heatmap: `data/artifacts/confusion_matrix.png`
