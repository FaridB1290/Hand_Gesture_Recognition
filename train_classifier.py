import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Evaluate on the test set
y_predict = model.predict(x_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

print('Classification Report:')
print(classification_report(y_test, y_predict))

# Cross-validation
cv_accuracy = np.mean(cross_val_score(model, data, labels, cv=5, scoring='accuracy'))
print('Cross-validated Accuracy: {:.2f}%'.format(cv_accuracy * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
