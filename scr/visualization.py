import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

def generate_ordinal_reports(predictions_output, test_df, label_encoder, model_name):
    y_pred = np.argmax(predictions_output.predictions, axis=1)
    y_true = predictions_output.label_ids
    target_names = ["Low", "Medium", "High"]

    # 1. طباعة البيانات الخام المستخدمة في الرسم
    df_results = pd.DataFrame({'Actual': [target_names[i] for i in y_true], 
                               'Predicted': [target_names[i] for i in y_pred]})
    print(f"\n{'='*60}\n📊 Raw Data for Predictions ({model_name})\n{'='*60}")
    print("Cross-Tabulation (Actual vs Predicted Count):")
    cross_tab = pd.crosstab(df_results['Actual'], df_results['Predicted'])
    print(cross_tab.to_markdown())
    print("\n")

    # 2. رسم مصفوفة الارتباك الترتيبية
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=target_names, yticklabels=target_names)
    
    # رسم خطوط توضح الخطأ الفادح (مثلاً توقع Low والواقع High)
    plt.plot([0, 3], [0, 3], color='black', lw=2, linestyle='--')
    
    plt.title(f"Ordinal Confusion Matrix ({model_name})\n(Off-diagonal elements represent prediction severity)", pad=15)
    plt.ylabel('Actual Processing Effort')
    plt.xlabel('Predicted Processing Effort')
    plt.tight_layout()
    plt.savefig(f'ordinal_cm_{model_name}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    # 3. حساب وطباعة توزع الأخطاء (Error Severity)
    errors = np.abs(y_true - y_pred)
    error_counts = pd.Series(errors).value_counts().sort_index()
    print(f"\n📉 Error Severity Breakdown (0=Correct, 1=1-Level Off, 2=2-Levels Off):")
    for severity, count in error_counts.items():
        print(f"  - Severity {severity}: {count} instances ({(count/len(y_true))*100:.1f}%)")
