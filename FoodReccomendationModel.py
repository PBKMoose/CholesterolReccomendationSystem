import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import os
import json
import joblib

file_path = "/content/processed_diet_data_corrected.csv"
df_food = pd.read_csv(file_path, low_memory=False)

nutrient_columns = [col for col in df_food.columns if col.isdigit()]
threshold_columns = [col for col in df_food.columns if "Threshold_" in col]
X_food = df_food[nutrient_columns + threshold_columns]

y_food_HDL = df_food["HDL_impact"]
y_food_LDL = df_food["LDL_impact"]
y_food_Total = df_food["Total_Cholesterol_impact"]

scaler = RobustScaler()
X_scaled_food = scaler.fit_transform(X_food)

X_train_food, X_test_food, y_HDL_train, y_HDL_test, y_LDL_train, y_LDL_test, y_Total_train, y_Total_test = train_test_split(
    X_scaled_food, y_food_HDL, y_food_LDL, y_food_Total, test_size=0.2, random_state=42, stratify=y_food_Total
)

y_HDL_train = y_HDL_train.to_numpy()
y_LDL_train = y_LDL_train.to_numpy()
y_Total_train = y_Total_train.to_numpy()
y_HDL_test = y_HDL_test.to_numpy()
y_LDL_test = y_LDL_test.to_numpy()
y_Total_test = y_Total_test.to_numpy()

classes = np.unique(y_Total_train)
class_weights_hdl = compute_class_weight('balanced', classes=classes, y=y_HDL_train)
class_weights_ldl = compute_class_weight('balanced', classes=classes, y=y_LDL_train)
class_weights_total = compute_class_weight('balanced', classes=classes, y=y_Total_train)

weights_hdl = tf.constant(class_weights_hdl, dtype=tf.float32)
weights_ldl = tf.constant(class_weights_ldl, dtype=tf.float32)
weights_total = tf.constant(class_weights_total, dtype=tf.float32)

def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        y_true_indices = tf.argmax(y_true, axis=1)
        weights = tf.gather(class_weights, y_true_indices)
        return cce * weights
    return loss

hdl_loss = weighted_categorical_crossentropy(weights_hdl)
ldl_loss = weighted_categorical_crossentropy(weights_ldl)
total_loss = weighted_categorical_crossentropy(weights_total)

def one_hot_encode(y):
    return tf.keras.utils.to_categorical(y + 1, num_classes=3)

y_HDL_train = one_hot_encode(y_HDL_train)
y_LDL_train = one_hot_encode(y_LDL_train)
y_Total_train = one_hot_encode(y_Total_train)

y_HDL_test = one_hot_encode(y_HDL_test)
y_LDL_test = one_hot_encode(y_LDL_test)
y_Total_test = one_hot_encode(y_Total_test)

def build_model(input_shape):
    input_layer = Input(shape=(input_shape,))
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.0005))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    HDL_output = Dense(3, activation="softmax", name="HDL_Output")(x)
    LDL_output = Dense(3, activation="softmax", name="LDL_Output")(x)
    Total_output = Dense(3, activation="softmax", name="Total_Output")(x)

    return Model(inputs=input_layer, outputs=[HDL_output, LDL_output, Total_output])

model = build_model(X_train_food.shape[1])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss={
        "HDL_Output": hdl_loss,
        "LDL_Output": ldl_loss,
        "Total_Output": total_loss,
    },
    metrics={
        "HDL_Output": "accuracy",
        "LDL_Output": "accuracy",
        "Total_Output": "accuracy",
    }
)

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train_food,
    {"HDL_Output": y_HDL_train, "LDL_Output": y_LDL_train, "Total_Output": y_Total_train},
    validation_data=(X_test_food, {"HDL_Output": y_HDL_test, "LDL_Output": y_LDL_test, "Total_Output": y_Total_test}),
    epochs=50,
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

test_results = model.evaluate(X_test_food, {"HDL_Output": y_HDL_test, "LDL_Output": y_LDL_test, "Total_Output": y_Total_test}, verbose=1)

print("\n Model Test Accuracy:")
print(f"HDL Accuracy: {test_results[4]:.4f}")
print(f"LDL Accuracy: {test_results[5]:.4f}")
print(f"Total Cholesterol Accuracy: {test_results[6]:.4f}")

y_pred = model.predict(X_test_food)
y_pred_HDL = np.argmax(y_pred[0], axis=1) - 1
y_pred_LDL = np.argmax(y_pred[1], axis=1) - 1
y_pred_Total = np.argmax(y_pred[2], axis=1) - 1

y_true_HDL = np.argmax(y_HDL_test, axis=1) - 1
y_true_LDL = np.argmax(y_LDL_test, axis=1) - 1
y_true_Total = np.argmax(y_Total_test, axis=1) - 1

comparison_df = pd.DataFrame({
    "Expected_HDL": y_true_HDL,
    "Predicted_HDL": y_pred_HDL,
    "Expected_LDL": y_true_LDL,
    "Predicted_LDL": y_pred_LDL,
    "Expected_Total": y_true_Total,
    "Predicted_Total": y_pred_Total
})

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
    plt.show()

def plot_conf_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["↓", "→", "↑"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
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

plot_conf_matrix(y_true_HDL, y_pred_HDL, "Confusion Matrix for HDL Adjustment", "conf_matrix_hdl")
plot_conf_matrix(y_true_LDL, y_pred_LDL, "Confusion Matrix for LDL Adjustment", "conf_matrix_ldl")
plot_conf_matrix(y_true_Total, y_pred_Total, "Confusion Matrix for Total Cholesterol Adjustment", "conf_matrix_total")

current_directory = os.getcwd()
food_model_dir = os.path.join(current_directory, 'models', 'food_model')
os.makedirs(food_model_dir, exist_ok=True)

model_path = os.path.join(food_model_dir, 'food_predictor.h5')
scaler_path = os.path.join(food_model_dir, 'food_scaler.joblib')
features_path = os.path.join(food_model_dir, 'food_features.json')

model.save(model_path)
joblib.dump(scaler, scaler_path)
with open(features_path, 'w') as f:
    json.dump({'feature_columns': X_food.columns.tolist()}, f)
