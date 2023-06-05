import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf

patients = pd.read_csv("data/heart_2020_cleaned.csv")
label = "HeartDisease"

print("\n10 first patients: ")
print(patients.head(10))

patients_labels = list(patients[label].unique())

patients[label] = patients[label].map(patients_labels.index)

print("\nLabels: ")
print(patients_labels)

np.random.seed(1)
# Use the ~10% of the examples as the testing set
# and the remaining ~90% of the examples as the training set.
test_indices = np.random.rand(len(patients)) < 0.1

train_ds = patients[~test_indices]
test_ds = patients[test_indices]

print("\nTraining examples: ", len(train_ds))

print("\nTesting examples: ", len(test_ds))


tf_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds, label=label)

model = tfdf.keras.CartModel()

model.fit(tf_train_ds)

model.compile("accuracy")
print("Train evaluation: ", model.evaluate(tf_train_ds, return_dict=True))

tf_test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds, label=label)
print("Test evaluation: ", model.evaluate(tf_test_ds, return_dict=True))

with open("plot.html", "w") as f: f.write(tfdf.model_plotter.plot_model(model, max_depth=10))

