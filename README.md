# heart_disease_analytics
PHÂN TÍCH VÀ DỰ ĐOÁN NGUY CƠ MẮC BỆNH SUY TIM 

> Dự án phân tích dữ liệu y tế và xây dựng mô hình Machine Learning dự đoán nguy cơ mắc bệnh tim sử dụng Apache Spark (PySpark), với trực quan hóa dữ liệu trên Power BI.

# Tổng quan dự án
  Bệnh tim mạch (CVDs) là nguyên nhân gây tử vong hàng đầu trên toàn cầu, cướp đi sinh mạng khoảng **17,9 triệu người mỗi năm**, chiếm 31% tổng số ca tử vong toàn thế giới. Trong đó, 4/5 trường hợp tử vong là do nhồi máu cơ tim và đột quỵ.
  Dự án này xây dựng một pipeline phân tích dữ liệu và học máy hoàn chỉnh nhằm:
- Phân tích các yếu tố lâm sàng ảnh hưởng đến nguy cơ mắc bệnh tim
- Xây dựng và so sánh các mô hình Machine Learning để dự đoán bệnh tim
- Tối ưu hóa siêu tham số với Optuna để cải thiện hiệu năng mô hình
- Trực quan hóa toàn bộ kết quả phân tích trên Power BI Dashboard

# Bộ dữ liệu
### Nguồn dữ liệu
  Bộ dữ liệu được tổng hợp từ **5 nguồn dữ liệu** bệnh tim nổi tiếng:

| Nguồn | Số mẫu |
|---|---|
| Cleveland | 303 |
| Hungarian | 294 |
| Switzerland | 123 |
| Long Beach VA | 200 |
| Stalog Dataset | 270 |
| **Tổng (sau loại trùng)** | **918** |

### Mô tả đặc trưng

| Đặc trưng | Mô tả | Kiểu dữ liệu |
|---|---|---|
| `Age` | Tuổi bệnh nhân (năm) | Số nguyên |
| `Sex` | Giới tính (M: Nam, F: Nữ) | Phân loại |
| `ChestPainType` | Loại đau ngực (TA, ATA, NAP, ASY) | Phân loại |
| `RestingBP` | Huyết áp khi nghỉ (mmHg) | Số nguyên |
| `Cholesterol` | Mức cholesterol (mg/dl) | Số nguyên |
| `FastingBS` | Đường huyết đói (1 nếu > 120 mg/dl) | Nhị phân |
| `RestingECG` | Kết quả điện tim khi nghỉ | Phân loại |
| `MaxHR` | Nhịp tim tối đa (60–202) | Số nguyên |
| `ExerciseAngina` | Đau ngực khi tập thể dục (Y/N) | Nhị phân |
| `Oldpeak` | Độ chênh ST khi gắng sức (mm) | Số thực |
| `ST_Slope` | Độ dốc đoạn ST (Up, Flat, Down) | Phân loại |
| `HeartDisease` | **Nhãn đầu ra** (1: Bệnh, 0: Bình thường) | Nhị phân |

### Thống kê tổng quan

- **Tổng số mẫu:** 918 bệnh nhân
- **Ca mắc bệnh tim:** 508 (55.34%)
- **Ca bình thường:** 410 (44.66%)
- **Tuổi trung bình:** 53.51 tuổi
- **Mất cân bằng dữ liệu:** Nhẹ — không cần xử lý oversampling

---

## Công nghệ sử dụng

```
Python 3.11
├── PySpark          — Xử lý dữ liệu phân tán & huấn luyện mô hình ML
├── Optuna           — Tối ưu hóa siêu tham số (Hyperparameter Tuning)
├── Pandas           — Tiền xử lý và phân tích dữ liệu
├── Seaborn          — Trực quan hóa dữ liệu (EDA)
└── Matplotlib       — Biểu đồ phân tích

Power BI Desktop     — Dashboard trực quan hóa 3 trang
Google Colab         — Môi trường thực thi
```

---

## Cấu trúc dự án

```
heart-disease-analytics/
│
├── data/
│   └── heart.csv                    # Bộ dữ liệu gốc
│
├── notebooks/
│   └── heart_disease_analysis.ipynb # Code ứng dụng Machine Learnin
│
├── powerbi/
│   └── heart disease.pbix  # File Power BI Dashboard
│
├── images/
│   └── Heart Disease Analytics Image # Hình ảnh các báo cáo trên Power BI
│
└── README.md
```

---

## Quy trình thực hiện

### 1. Khởi tạo môi trường Spark

```python
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("HeartDiseaseSpark") \
    .getOrCreate()
```

### 2. Phân tích khám phá dữ liệu (EDA)

Phân tích phân phối dữ liệu theo từng đặc trưng so với biến mục tiêu `HeartDisease`:

**Phát hiện chính từ EDA:**

- **ST_Slope:** Bệnh nhân có `Flat` slope chiếm tỷ lệ mắc bệnh tim cao nhất
- **ChestPainType:** Nhóm `ASY` (không triệu chứng) có 392/508 ca bệnh — nguy hiểm vì không có dấu hiệu rõ ràng
- **MaxHR:** Người mắc bệnh tim có nhịp tim tối đa thấp hơn rõ rệt
- **Oldpeak:** Giá trị cao hơn ở nhóm bệnh — dấu hiệu thiếu máu cục bộ (ischemia)
- **Age:** Nguy cơ tăng rõ theo độ tuổi, đỉnh tập trung 55–65 tuổi
- **RestingBP & Cholesterol:** Không có sự khác biệt rõ rệt giữa 2 nhóm

### 3. Tiền xử lý dữ liệu

```python
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

# Mã hóa biến phân loại
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG',
                    'ExerciseAngina', 'ST_Slope']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index")
            for col in categorical_cols]

# Chuẩn hóa biến số
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler(inputCol="numerical_feature",
                        outputCol="scaled_feature")

# Ghép đặc trưng thành vector
assembler = VectorAssembler(inputCols=all_features, outputCol="features")
```

### 4. Huấn luyện & So sánh mô hình

Ba mô hình được huấn luyện và so sánh trên cùng tập test (80/20 split, `seed=42`):

```python
models = [
    LogisticRegression(labelCol="label", featuresCol="features"),
    RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100),
    GBTClassifier(labelCol="label", featuresCol="features")
]
```

### 5. Tối ưu hóa siêu tham số với Optuna

Optuna được sử dụng để tối ưu hóa Random Forest với **50 trials**, tối đa hóa chỉ số **Recall** — vì trong bài toán y tế, bỏ sót bệnh nhân (False Negative) nguy hiểm hơn báo nhầm (False Positive):

```python
def objective(trial):
    rf = RandomForestClassifier(
        numTrees=trial.suggest_int("numTrees", 50, 300, step=50),
        maxDepth=trial.suggest_int("maxDepth", 2, 30, step=2),
        labelCol="label",
        featuresCol="features",
        seed=42
    )
    # Tối đa hóa Weighted Recall
    ...
```

**Siêu tham số tốt nhất tìm được:**
```
numTrees = 100
maxDepth = 26
Best Recall Score = 0.8993
```

---

## Kết quả

### So sánh các mô hình (trước tối ưu hóa)

| Mô hình | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.8523 | 0.8555 | 0.8523 | 0.8514 |
| Gradient Boosting | 0.8658 | 0.8675 | 0.8658 | 0.8652 |
| **Random Forest** | **0.8725** | **0.8798** | **0.8725** | **0.8712** |

### Random Forest — Trước và sau tối ưu hóa Optuna

| Chỉ số | Before Tuning | After Tuning | Cải thiện |
|---|---|---|---|
| Accuracy | 0.8725 | **0.8993** | +2.68% |
| Precision | 0.8798 | **0.9007** | +2.09% |
| Recall | 0.8725 | **0.8993** | +2.68% |
| F1 Score | 0.8712 | **0.8990** | +2.78% |

### Feature Importance — Random Forest (After Tuning)

| Thứ hạng | Đặc trưng | Độ quan trọng |
|---|---|---|
| 1 | ST_Slope | 0.310 |
| 2 | ChestPainType | 0.181 |
| 3 | ExerciseAngina | 0.119 |
| 4 | Oldpeak | 0.106 |
| 5 | Cholesterol | 0.089 |
| 6 | MaxHR | 0.080 |
| 7 | Age | 0.041 |
| 8 | Sex | 0.034 |
| 9 | RestingBP | 0.032 |
| 10 | RestingECG | 0.010 |

> **ST_Slope là đặc trưng quan trọng nhất** với độ quan trọng 0.310 — gấp 1.7 lần đặc trưng quan trọng thứ 2 (ChestPainType).

### Tại sao Recall là chỉ số ưu tiên?

$$\text{Recall} = \frac{TP}{TP + FN}$$

Trong bài toán chẩn đoán bệnh tim:
- **FN (False Negative):** Dự đoán bệnh nhân không mắc bệnh nhưng thực tế có bệnh → bỏ lỡ cơ hội điều trị sớm → **nguy hiểm đến tính mạng**
- **FP (False Positive):** Dự đoán có bệnh nhưng thực tế không → bệnh nhân làm thêm xét nghiệm → **ít gây hại hơn**

Do đó, tối đa hóa Recall để giảm thiểu FN là ưu tiên số 1 trong bài toán y tế này.

---

## Trực quan hóa bằng Power BI

Dashboard được xây dựng trên **Power BI Desktop** gồm 3 trang:

### Trang 1 — Tổng quan & Phân tích dữ liệu
- 4 KPI Cards: Tổng bệnh nhân, ca bệnh, ca bình thường, tuổi trung bình
- Donut Chart: Tỷ lệ mắc bệnh tim
- Clustered Bar: Phân bố theo giới tính và kiểu đau ngực
- Histogram: Phân bố theo độ tuổi
- Slicer: Bộ lọc tương tác theo giới tính

### Trang 2 — Phân tích đặc trưng lâm sàng
- Histogram: Phân phối MaxHR và Oldpeak theo tình trạng bệnh
- Bar Chart: ST_Slope, RestingECG, ExerciseAngina, FastingBS vs Bệnh tim
- Scatter Chart: Tuổi vs MaxHR và Tuổi vs Oldpeak (phân màu theo HeartDisease)

### Trang 3 — Kết quả mô hình Machine Learning
- KPI Cards: Mô hình tốt nhất, Recall, Precision, F1-Score
- Clustered Bar: So sánh 3 mô hình × 4 chỉ số đánh giá
- Bar Chart: Feature Importance với gradient màu theo mức độ quan trọng
- Before/After Chart: So sánh Random Forest trước và sau tối ưu Optuna

---

## Hướng dẫn chạy

### Yêu cầu môi trường

```bash
pip install pyspark findspark optuna pandas seaborn matplotlib
```

### Chạy trên Google Colab

1. Upload file `heart.csv` lên Colab
2. Mở file `heart_disease_analysis.ipynb`
3. Chạy tuần tự các cell từ trên xuống dưới
4. Kết quả mô hình và biểu đồ sẽ hiển thị inline

### Xem Power BI Dashboard

1. Tải và cài đặt **Power BI Desktop** (miễn phí)
2. Mở file `HeartDisease_Dashboard.pbix`
3. Kết nối lại nguồn dữ liệu nếu được yêu cầu: trỏ đến file `heart.csv`

---

## Kết luận

**Mô hình được chọn:** Random Forest sau tối ưu hóa Optuna

**Lý do lựa chọn:**
- Đạt Recall cao nhất (**89.93%**) — giảm thiểu tối đa bỏ sót bệnh nhân
- Accuracy **89.93%** và Precision **90.07%** — cân bằng tốt giữa các chỉ số
- Cải thiện đáng kể so với baseline sau tối ưu hóa

**Phát hiện quan trọng từ phân tích:**
- `ST_Slope` là yếu dự báo mạnh nhất, đặc trưng quan trọng nhất
- Bệnh nhân `ASY` (không có triệu chứng đau ngực) chiếm đa số ca bệnh tim —> nhấn mạnh tầm quan trọng của tầm soát định kỳ
- Nguy cơ mắc bệnh tim tăng rõ từ độ tuổi 50 trở lên

**Hướng phát triển tiếp theo:**
- Thử nghiệm thêm các mô hình Machine Learning khác
- Áp dụng SMOTE nếu mất cân bằng dữ liệu tăng
- Triển khai mô hình thành REST API phục vụ hệ thống cảnh báo sớm
- Mở rộng tích hợp dữ liệu thời gian thực từ thiết bị đeo tay

---

## Tác giả

Dự án được thực hiện như một bài tập phân tích dữ liệu thực tế, kết hợp kỹ năng:
- **Data Analysis:** EDA, thống kê mô tả, phân tích tương quan
- **Big Data:** Xử lý dữ liệu với Apache Spark (PySpark)
- **Machine Learning:** Classification, Feature Engineering, Hyperparameter Tuning
- **Data Visualization:** Power BI Dashboard (3 trang, đơn sắc xanh)

---

Dự án sử dụng bộ dữ liệu công khai từ [Kaggle — Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) theo giấy phép Open Database License.

---

