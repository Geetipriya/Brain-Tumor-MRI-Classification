import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def evaluate_predictions(pred_csv, model_name, class_names):
    df = pd.read_csv(pred_csv)
    print(f"=== Results for {model_name} ===")
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        y_true = split_df['true_class']
        y_pred = split_df['predicted_class'].astype(str)
        acc = (y_true == y_pred).mean()
        print(f"{split.capitalize()} Accuracy: {acc:.2%}")
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(xticks_rotation='vertical')
        plt.title(f'{model_name} {split.capitalize()} Confusion Matrix')
        plt.show()
        print(classification_report(y_true, y_pred, labels=class_names))

# Usage example:
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
evaluate_predictions('predictions_custom.csv', 'Custom CNN', class_names)
evaluate_predictions('predictions_resnet.csv', 'ResNet50', class_names)