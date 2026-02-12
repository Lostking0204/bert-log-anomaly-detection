# bert-log-anomaly-detection
BERT-Log implementation for system log anomaly detection


## Performance Comparison
Following the BERT-Log architecture (Chen & Liao, 2022), our implementation achieved the following results on the HDFS dataset:

| Metric | Original Paper | Our Implementation (Feb 12) |
| :--- | :--- | :--- |
| **Accuracy** | ~0.998 | **0.9990** |
| **Precision** | 0.996 | **0.9659** |
| **Recall** | 0.996 | **1.0000** |
| **F1-Score** | **0.9930** | **0.9827** |

> **Note:** Our implementation reached a perfect **1.0000 Recall** after 17,971 training batches, ensuring all anomalies were successfully detected.

