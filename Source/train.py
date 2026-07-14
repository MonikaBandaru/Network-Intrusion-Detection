import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train,y_train,X_test,y_test = joblib.load(
    "Models/preprocessed.pkl"
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

print("Accuracy")

print(
    accuracy_score(
        y_test,
        pred
    )
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        pred
    )
)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        pred
    )
)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("Images/Confusion_Matrix.png")
plt.close()

joblib.dump(
    model,
    "Models/model.pkl"
)

print("\nModel Saved")