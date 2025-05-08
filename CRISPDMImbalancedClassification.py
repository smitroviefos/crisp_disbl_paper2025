import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.contingency_tables import cochrans_q
from collections import defaultdict
from sklearn.base import clone

class CRISPDMImbalancedClassification:
    def __init__(self, dataset_path, target_column, sampler='SMOTE'):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.sampler_name = sampler
        self.sampler = SMOTE(random_state=42) if sampler == 'SMOTE' else ADASYN(random_state=42)
        self.model = RandomForestClassifier(random_state=42)

    def business_understanding(self):
        print("Phase: Business Understanding")
        print("Objective: Identify a solution for an imbalanced classification problem using the CRISP-DM methodology.\n")

    def data_understanding(self):
        print("Phase: Data Understanding")
        self.df = pd.read_csv(self.dataset_path)
        print("Dataset shape:", self.df.shape)
        print("Class distribution:\n", self.df[self.target_column].value_counts())
        print("\nDescriptive statistics:\n", self.df.describe())

         # Visualize class imbalance
        sns.countplot(x=self.target_column, data=self.df)
        plt.title(f"Class Distribution - {self.dataset_path}")
        plt.tight_layout()
        plt.show()

    def data_preparation(self):
        print("Phase: Data Preparation")
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # Encode labels if needed
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)
        self.class_names = np.unique(y)

        # Train/test split and scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def modeling(self):
        print(f"Phase: Modeling ({self.sampler_name})")
        X_resampled, y_resampled = self.sampler.fit_resample(self.X_train, self.y_train)
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled

        start_time = time.time()
        self.model.fit(X_resampled, y_resampled)
        self.training_time = time.time() - start_time
        print(f"Model trained in {round(self.training_time, 2)} seconds.")

    def evaluation(self):
        print("Phase: Evaluation")
        y_pred = self.model.predict(self.X_test)
        y_score = self.model.predict_proba(self.X_test)

        f1 = f1_score(self.y_test, y_pred, average=None)
        bal_acc = balanced_accuracy_score(self.y_test, y_pred)

        # AUC-ROC
        if len(self.class_names) == 2:
            # Binary classification 
            auc = roc_auc_score(self.y_test, y_score[:, 1])
        else:
            # Multiclass classification
            y_test_bin = label_binarize(self.y_test, classes=self.class_names)
            auc = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")

        print("\nClassification report:\n", classification_report(self.y_test, y_pred))
        print("Balanced Accuracy:", round(bal_acc, 4))
        print("F1-score per class:", np.round(f1, 4))
        print("AUC-ROC:", round(auc, 4))

        self.conf_matrix = confusion_matrix(self.y_test, y_pred)

    def deployment(self):
        print("Phase: Deployment")
        sns.heatmap(self.conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({self.sampler_name} - {self.dataset_path})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        print("Model ready for evaluation or integration.\n")
    
    def save_results(self, dataset_name, results_file="crispdm_evaluation.csv",
                     other_model=None, other_sampler_name=None, p_value=None, test_name=None):
        """
        Save evaluation results and (optionally) statistical validation results to CSV.
        """
        y_pred = self.model.predict(self.X_test)
        y_score = self.model.predict_proba(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average=None)
        bal_acc = balanced_accuracy_score(self.y_test, y_pred)

        # For AUC-ROC
        if len(self.class_names) == 2:
            auc = roc_auc_score(self.y_test, y_score[:, 1])
        else:
            y_test_bin = label_binarize(self.y_test, classes=self.class_names)
            auc = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")

        result = {
            "Dataset": dataset_name,
            "Sampler": self.sampler_name,
            "Training Time (s)": round(self.training_time, 4),
            "Balanced Accuracy": round(bal_acc, 4),
            "AUC-ROC": round(auc, 4),
        }

        for i, score in enumerate(f1):
            result[f"F1_Class_{i}"] = round(score, 4)

        if other_model and p_value is not None:
            result["Compared With"] = other_sampler_name
            result["Statistical Test"] = test_name
            result["P-Value"] = round(p_value, 4)
            result["Significant Difference"] = "YES" if p_value < 0.05 else "NO"
        else:
            result["Compared With"] = ""
            result["Statistical Test"] = ""
            result["P-Value"] = ""
            result["Significant Difference"] = ""

        df_result = pd.DataFrame([result])
        try:
            df_existing = pd.read_csv(results_file)
            df_result = pd.concat([df_existing, df_result], ignore_index=True)
        except FileNotFoundError:
            pass

        df_result.to_csv(results_file, index=False)
        print(f"Results saved to:: {results_file}")

    def statistical_validation(self, other_model, other_sampler_name="OTHER"):
        """
        Perform statistical validation using McNemar test (binary) or Cochran's Q test (multiclass).
        Returns: test_name (str), p_value (float)
        """
        print(f"\nStatistical validation: {self.sampler_name} vs. {other_sampler_name}")

        y1_pred = self.model.predict(self.X_test)
        y2_pred = other_model.predict(self.X_test)

        # Binary classification → McNemar
        if len(np.unique(self.y_test)) == 2:
            b = np.sum((y1_pred == self.y_test) & (y2_pred != self.y_test))  # correct/true 1., incorrect/false 2.
            c = np.sum((y1_pred != self.y_test) & (y2_pred == self.y_test))  # correct/true 2., incorrect/false 1.

            table = [[0, b],
                     [c, 0]]

            result = mcnemar(table, exact=True)
            print(f"McNemar test (p-value): {result.pvalue:.4f}")
            if result.pvalue < 0.05:
                print("Statistically significant difference detected (p < 0.05)")
            else:
                print("No statistically significant difference (p ≥ 0.05)")

        # Multiclass classification → Cochranov Q test
        else:
            correct1 = (y1_pred == self.y_test).astype(int)
            correct2 = (y2_pred == self.y_test).astype(int)

            data = np.vstack([correct1, correct2]).T  # shape (n_samples, n_models)

            if data.shape[1] < 2 or data.shape[0] < 10:
                print("Insufficient data for statistical test (minimum 10 samples required)")
                return

            q_stat, p_value = cochranq(data)
            print(f"Cochran’s Q test (p-value): {p_value:.4f}")
            if p_value < 0.05:
                print("Statistically significant difference detected (p < 0.05)")
            else:
                print("No statistically significant difference (p ≥ 0.05)")
    
        # Povrat vrijednosti
        if len(np.unique(self.y_test)) == 2:
            return "McNemar", result.pvalue
        else:
            return "CochranQ", p_value

    def plot_statistical_tests(results_file="crispdm_evaluation.csv"):
        """
        Visualize p-values from statistical tests across datasets and samplers.
        """
        try:
            df = pd.read_csv(results_file)
        except FileNotFoundError:
            print("Evaluation file not found.")
            return

        df_tests = df[df["P-Value"].notnull() & (df["P-Value"] != "")]
        if df_tests.empty:
            print("No statistical test results available for visualization.")
            return

        df_tests["P-Value"] = pd.to_numeric(df_tests["P-Value"])
        df_tests["Label"] = df_tests["Dataset"] + " (" + df_tests["Sampler"] + " vs. " + df_tests["Compared With"] + ")"

        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_tests, x="Label", y="P-Value", hue="Significant Difference", dodge=False)
        plt.axhline(0.05, ls="--", color="red", label="α = 0.05")
        plt.title("P-values from statistical tests across datasets")
        plt.ylabel("P-Value")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Statistical Significance", loc="upper right")
        plt.tight_layout()
        plt.show()

    def write_log(self, dataset_name, log_file="crispdm_report.log",
                  other_sampler_name=None, p_value=None, test_name=None):
        """
         Write detailed log of results and test statistics.
        """
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Sampler: {self.sampler_name}\n")
            f.write(f"Training time: {round(self.training_time, 4)} seconds\n")

            # Metrike
            y_pred = self.model.predict(self.X_test)
            y_score = self.model.predict_proba(self.X_test)
            f1 = f1_score(self.y_test, y_pred, average=None)
            bal_acc = balanced_accuracy_score(self.y_test, y_pred)

            if len(self.class_names) == 2:
                auc = roc_auc_score(self.y_test, y_score[:, 1])
            else:
                y_test_bin = label_binarize(self.y_test, classes=self.class_names)
                auc = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")

            f.write(f"Balanced Accuracy: {round(bal_acc, 4)}\n")
            f.write(f"AUC-ROC: {round(auc, 4)}\n")
            for i, score in enumerate(f1):
                f.write(f"F1-score for class  {i}: {round(score, 4)}\n")

            # Statistika
            if p_value is not None and test_name:
                f.write(f"Statistical Test: {test_name}, P-value: {round(p_value, 4)}\n")
                interpretation = "Significant difference" if p_value < 0.05 else "No significant difference"
                f.write(f"Interpretation: {interpretation}\n")

            f.write("-" * 50 + "\n")

# ===============================
# Automatisation for multiple datasets
# ===============================

if __name__ == "__main__":
    datasets = [
        {"path": "creditcard.csv", "target": "Class"},
        {"path": "phishing.csv", "target": "Result"}
    ]

    samplers = ["SMOTE", "ADASYN"]
    # Step 1: Run and store models
    results = {}

    for ds in datasets:
        for sampler in samplers:
            print(f"\n\n================= Processing dataset: {ds['path']} | Sampler: {sampler} =================")
            crisp = CRISPDMImbalancedClassification(
                dataset_path=ds["path"],
                target_column=ds["target"],
                sampler=sampler
            )

            crisp.business_understanding()
            crisp.data_understanding()
            crisp.data_preparation()
            crisp.modeling()
            crisp.evaluation()
            crisp.deployment()
            crisp.save_results(dataset_name=ds["path"])

            # Store model instance for statistical comparison
            results[(ds["path"], sampler)] = crisp
            
    # Step 2: Perform statistical validation and save results
        for ds in datasets:
            key_smote = (ds["path"], "SMOTE")
            key_adasyn = (ds["path"], "ADASYN")
            
            if key_smote in results and key_adasyn in results:
                smote_model = results[key_smote]
                adasyn_model = results[key_adasyn]

                print(f"\n=== Statistical validation for dataset: {ds['path']} ===")
                test_name, p_value = smote_model.statistical_validation(
                    other_model=adasyn_model.model,
                    other_sampler_name="ADASYN"
                )

                # Save results including statistical test for SMOTE model
                smote_model.save_results(
                    dataset_name=ds["path"],
                    other_model=adasyn_model.model,
                    other_sampler_name="ADASYN",
                    p_value=p_value,
                    test_name=test_name
                )
                smote_model.write_log(
                    dataset_name=ds["path"],
                    other_sampler_name="ADASYN",
                    p_value=p_value,
                    test_name=test_name
                )
        # Step 3: Visualize p-values across datasets
        CRISPDMImbalancedClassification.plot_statistical_tests()


