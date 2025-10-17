import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import os
import json
import joblib

current_directory = os.getcwd()
os.makedirs(os.path.join(current_directory, 'models', 'cholesterol_model'), exist_ok=True)

synthetic_data_path = "/content/augmented_dataset.csv"
df_synthetic = pd.read_csv(synthetic_data_path, low_memory=False)
df_synthetic.fillna(0, inplace=True)

for col in ["HDL_Adjustment", "LDL_Adjustment", "Total_Cholesterol_Adjustment"]:
    df_synthetic[col] = pd.to_numeric(df_synthetic[col], errors='coerce').fillna(0).astype(int)

df_synthetic["LDL_to_HDL_Ratio"] = df_synthetic["LDL_Level"] / (df_synthetic["HDL_Level"] + 1)
df_synthetic["Cholesterol_Difference"] = df_synthetic["Total_Cholesterol"] - df_synthetic["LDL_Level"]

df_synthetic['combined_target'] = df_synthetic.apply(
    lambda row: f"{int(row['HDL_Adjustment'])}_{int(row['LDL_Adjustment'])}_{int(row['Total_Cholesterol_Adjustment'])}", axis=1
)

X = df_synthetic.drop(columns=['combined_target', 'HDL_Adjustment', 'LDL_Adjustment', 'Total_Cholesterol_Adjustment'])
y_combined = df_synthetic['combined_target']

class_counts = df_synthetic['combined_target'].value_counts()
minority_classes = class_counts[class_counts < class_counts.max()]
if len(minority_classes) > 0:
    min_minority_size = minority_classes.min()
    if min_minority_size <= 1:
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_combined_resampled = ros.fit_resample(X, y_combined)
    else:
        k_neighbors = min(5, min_minority_size - 1)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_combined_resampled = smote.fit_resample(X, y_combined)
else:
    X_resampled, y_combined_resampled = X, y_combined

y_resampled_split = y_combined_resampled.str.split('_', expand=True).astype(int)
y_HDL_resampled = y_resampled_split[0]
y_LDL_resampled = y_resampled_split[1]
y_Total_resampled = y_resampled_split[2]

df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["HDL_Adjustment"] = y_HDL_resampled
df_resampled["LDL_Adjustment"] = y_LDL_resampled
df_resampled["Total_Cholesterol_Adjustment"] = y_Total_resampled

num_classes = 3
y_HDL_encoded = to_categorical(df_resampled["HDL_Adjustment"] + 1, num_classes=num_classes)
y_LDL_encoded = to_categorical(df_resampled["LDL_Adjustment"] + 1, num_classes=num_classes)
y_Total_encoded = to_categorical(df_resampled["Total_Cholesterol_Adjustment"] + 1, num_classes=num_classes)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_resampled[X.columns])

X_train, X_test, y_HDL_train, y_HDL_test, y_LDL_train, y_LDL_test, y_Total_train, y_Total_test = train_test_split(
    X_scaled, y_HDL_encoded, y_LDL_encoded, y_Total_encoded, test_size=0.2, random_state=42
)

np.random.seed(42)
noise = np.random.normal(0, 0.1, y_HDL_train.shape)
y_HDL_train = np.clip(y_HDL_train + noise, 0, 1)

input_layer = Input(shape=(X_train.shape[1],))
x = GaussianNoise(0.1)(input_layer)
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

hdl_output = Dense(num_classes, activation="softmax", name="HDL_Output")(x)
ldl_output = Dense(num_classes, activation="softmax", name="LDL_Output")(x)
total_output = Dense(num_classes, activation="softmax", name="Total_Output")(x)

adjustment_model = Model(inputs=input_layer, outputs=[hdl_output, ldl_output, total_output])
adjustment_model.compile(
    optimizer=Adam(learning_rate=5e-5, clipvalue=1.0),
    loss={
        "HDL_Output": "categorical_crossentropy",
        "LDL_Output": "categorical_crossentropy",
        "Total_Output": "categorical_crossentropy",
    },
    loss_weights={"HDL_Output": 0.1, "LDL_Output": 1.0, "Total_Output": 2.5},
    metrics={
        "HDL_Output": ["accuracy"],
        "LDL_Output": ["accuracy"],
        "Total_Output": ["accuracy"]
    }
)

early_stopping = EarlyStopping(monitor="val_Total_Output_accuracy", patience=5, restore_best_weights=True, mode="max")
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

history = adjustment_model.fit(
    X_train,
    {"HDL_Output": y_HDL_train, "LDL_Output": y_LDL_train, "Total_Output": y_Total_train},
    validation_data=(X_test, {"HDL_Output": y_HDL_test, "LDL_Output": y_LDL_test, "Total_Output": y_Total_test}),
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]
)

model_path = os.path.join(current_directory, 'models', 'cholesterol_model', 'cholesterol_predictor.h5')
scaler_path = os.path.join(current_directory, 'models', 'cholesterol_model', 'cholesterol_scaler.joblib')
features_path = os.path.join(current_directory, 'models', 'cholesterol_model', 'cholesterol_features.json')

adjustment_model.save(model_path)
joblib.dump(scaler, scaler_path)
with open(features_path, 'w') as f:
    json.dump({'feature_columns': X.columns.tolist()}, f)

y_pred = adjustment_model.predict(X_test)
y_pred_HDL = np.argmax(y_pred[0], axis=1) - 1
y_pred_LDL = np.argmax(y_pred[1], axis=1) - 1
y_pred_Total = np.argmax(y_pred[2], axis=1) - 1

y_true_HDL = np.argmax(y_HDL_test, axis=1) - 1
y_true_LDL = np.argmax(y_LDL_test, axis=1) - 1
y_true_Total = np.argmax(y_Total_test, axis=1) - 1

def plot_metric(history, metric, output_name, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[f'{output_name}_{metric}'], label=f'Training {ylabel}')
    plt.plot(history.history[f'val_{output_name}_{metric}'], label=f'Validation {ylabel}')
    plt.title(f'{title} for {output_name}')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.show()

plot_metric(history, 'accuracy', 'HDL_Output', 'Accuracy', 'Accuracy', 'hdl_accuracy')
plot_metric(history, 'accuracy', 'LDL_Output', 'Accuracy', 'Accuracy', 'ldl_accuracy')
plot_metric(history, 'accuracy', 'Total_Output', 'Accuracy', 'Accuracy', 'total_accuracy')

plot_metric(history, 'loss', 'HDL_Output', 'Loss', 'Loss', 'hdl_loss')
plot_metric(history, 'loss', 'LDL_Output', 'Loss', 'Loss', 'ldl_loss')
plot_metric(history, 'loss', 'Total_Output', 'Loss', 'Loss', 'total_loss')

print("HDL Classification Report:\n", classification_report(y_true_HDL, y_pred_HDL, digits=4))
print("LDL Classification Report:\n", classification_report(y_true_LDL, y_pred_LDL, digits=4))
print("Total Cholesterol Classification Report:\n", classification_report(y_true_Total, y_pred_Total, digits=4))

def plot_conf_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["↓", "→", "↑"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.savefig(f"{filename}.png")
    plt.show()

plot_conf_matrix(y_true_HDL, y_pred_HDL, "Confusion Matrix for HDL Adjustment", "conf_matrix_hdl")
plot_conf_matrix(y_true_LDL, y_pred_LDL, "Confusion Matrix for LDL Adjustment", "conf_matrix_ldl")
plot_conf_matrix(y_true_Total, y_pred_Total, "Confusion Matrix for Total Cholesterol Adjustment", "conf_matrix_total")
