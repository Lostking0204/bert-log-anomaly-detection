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


## Experimental Results

### **Performance Comparison**
The following table compares our implementation (trained for 5 epochs on the full HDFS dataset) against the baseline results reported in the original BERT-Log paper.

| Metric | Original Paper (Baseline) | Our Implementation (5 Epochs) | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.9980 | **0.9997** | 📈 Improved |
| **Precision** | **0.9960** | 0.9899 | 📉 Slight Drop |
| **Recall** | 0.9960 | **0.9997** | 📈 Improved |
| **F1-Score** | 0.9930 | **0.9948** | 📈 Improved |

### **Key Observations**
- **High Sensitivity:** Our model achieved a **Recall of 0.9997**, significantly reducing False Negatives which is crucial for anomaly detection.
- **Better Balance:** The **F1-Score of 0.9948** proves that our model provides a superior harmonic mean between Precision and Recall compared to the original research.
- **Robustness:** These results were validated on a standard **20% test split (115,012 sequences)**, ensuring high generalizability.

