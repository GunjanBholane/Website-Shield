from sklearn.svm import OneClassSVM

# Assume 'traffic_data' is your dataset containing network traffic information
X_train = traffic_data['normal']  # normal traffic
X_test = traffic_data['suspicious']  # traffic to check for attacks

# Train the model
model = OneClassSVM(gamma='auto').fit(X_train)

# Predict on new data
y_pred = model.predict(X_test)

# 1 means normal, -1 means abnormal (potential DDoS attack)
print(y_pred)
