# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adamax
from tensorflow import keras
import seaborn as sns
print(sns.__version__)

df = pd.read_csv('2022_2025_city_of_london_street.csv')
df.head()

# Drop duplicates rows
df.drop_duplicates(inplace=True)

# Drop the Crime ID, source column, and source file columns since they contain duplicate information/administrative purposes when merging the datasets, or are identifiers.
df.drop(columns=['Crime ID', 'Context', 'source_file', 'source_folder'], inplace=True)

# Strip column names
# REASON: In some cases, when reading CSV files, extra spaces can be inadvertently added in the column names.
# This can lead to issues when trying to access these columns later in the code, as the names won't match exactly.
df.columns = df.columns.str.strip()


# Fill missing numeric values using KNN imputation, which in this case is the GPS coordinates (Longitude, Latitude)
numeric_cols = ['Longitude', 'Latitude']
imputer = KNNImputer(n_neighbors=5)
# Note: For production, fit the imputer on the training set only to avoid data leakage.
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Encode target values, so that the crime ('Crime type') can be used for classification, (LABELS) to (NUMERICAL RANGE)
target_col = 'Crime type'
encoder = LabelEncoder()

df['target_encoded'] = encoder.fit_transform(df[target_col])
num_classes = df['target_encoded'].nunique()

# Features: numeric and categorical
categorical_cols = ['Reported by', 'Falls within', 'LSOA code', 'LSOA name']
df_encoded = pd.get_dummies(df[categorical_cols])

X = pd.concat([df[numeric_cols], df_encoded], axis=1).values
y = to_categorical(df['target_encoded'], num_classes=num_classes)

# Standardize numeric features
scaler = StandardScaler()
X[:, :len(numeric_cols)] = scaler.fit_transform(X[:, :len(numeric_cols)])

X = X.astype("float32")  # ensure proper dtype

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

 
model = Sequential()
model.add(Input(X[1].shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(10))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train,y_train,verbose=0,epochs=128)
model.summary()
pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1) 
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

#path to where the file will be saved
save_path = "Model"

# save neural network structure to JSON (no weights)
model_json = model.to_json()
with open(os.path.join(save_path,"network.json"), "w") as json_file:
    json_file.write(model_json)

# save entire network to HDF5 (save everything, suggested)
model.save(os.path.join(save_path,"network.h5"))

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=512
)

#make predictions (will give a probability distribution)
pred_hot = model.predict(X_test)
#now pick the most likely outcome
pred = np.argmax(pred_hot,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#calculate accuracy
score = metrics.accuracy_score(y_compare, pred)

print("Accuracy score: {}".format(score))

print(pred_hot[:5])
print(pred)

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification Report
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))

mat = confusion_matrix(pred, y_compare)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.show()