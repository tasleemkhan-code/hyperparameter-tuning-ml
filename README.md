⚙️ Hyperparameter Tuning: GridSearchCV vs RandomizedSearchCV

Tuning an SVM classifier on the Iris dataset with GridSearchCV and RandomizedSearchCV (Scikit-learn), and comparing the two search strategies.

Open In Colab

📌 Overview

Model settings (hyperparameters) are not learned from data, so they have to be searched for. This project compares two common ways to do that:

Method	How it works
GridSearchCV	Tries every combination in the parameter grid (exhaustive)
RandomizedSearchCV	Tries a fixed number of randomly picked combinations (faster on large grids)
📊 Dataset
Source: Iris dataset from Seaborn (sns.load_dataset("iris"))
150 samples, 4 features (sepal_length, sepal_width, petal_length, petal_width)
Target: species (setosa, versicolor, virginica)
🔧 Workflow
Load the Iris dataset and split into features (x) and target (y)
Train/test split (67/33, random_state=42)
Train baseline models: KNN (n_neighbors=13) and SVM (SVC(gamma='auto'))
Run GridSearchCV on SVM (5-fold cross-validation)
Run RandomizedSearchCV on SVM (n_iter=4, 5-fold cross-validation)
Compare the cross-validation scores of every tried combination
✅ Results
Baseline (single train/test split)
Model	Test Accuracy
KNN (k=13)	1.00
SVM (default C=1, rbf kernel)	1.00
GridSearchCV (SVM, 5-fold CV)

Parameter grid: C in [1, 10, 20] and kernel in [rbf, linear] → 6 combinations.

C	Kernel	Mean CV Score
1	rbf	0.9800
1	linear	0.9800
10	rbf	0.9800
10	linear	0.9733
20	rbf	0.9667
20	linear	0.9667

Best mean CV score: 0.98 (three combinations tied: C=1 rbf, C=1 linear, C=10 rbf).

RandomizedSearchCV (SVM, 5-fold CV)

Search space: C in [1, 10, 20, 30, 40, 50] and kernel in [rbf, linear] → 12 possible combinations, 4 sampled.

C	Kernel	Mean CV Score
40	linear	0.9600
1	linear	0.9800
50	rbf	0.9600
40	rbf	0.9600

Best mean CV score: 0.98 (C=1, linear kernel). The sampled combinations differ on every run because no random_state was set.

🔍 Key Findings
Both searches reached the same best CV score (0.98), and RandomizedSearchCV did it by trying only 4 of the 12 possible combinations.
Smaller C values worked better here. In the grid search, C=1 and C=10 (rbf) scored 0.98, while C=20 dropped to 0.9667.
Tuning did not beat the baseline. The default SVM already scored 100% on the test split. Iris is a small, easy dataset, so there is little room for improvement. Note that the baseline test accuracy (one split of 50 samples) and the CV scores (5 folds on all 150 samples) are measured differently and are not directly comparable.
The scores are very close (0.9667 to 0.98 across all tried settings), so on a dataset this small the differences are within normal variation.
🛠️ Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

▶️ How to Run

Google Colab: click the "Open in Colab" badge above and run all cells.

Locally:

bash
git clone https://github.com/tasleemkhan-code/hyperparameter-tuning-ml.git
cd hyperparameter-tuning-ml
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook HyperParameterTuning.ipynb

👤 Author

Tasleem Khan BCA (Hons.) Student, Cloud & AI/ML GitHub: @tasleemkhan-code
