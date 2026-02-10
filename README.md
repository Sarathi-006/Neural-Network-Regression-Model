# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model
![ஸ்கிரீன்ஷாட் 2025-03-03 114257](https://github.com/user-attachments/assets/70ab278e-36ea-44f7-b6b7-3a7f7f44fdb5)




## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: PARTHASARATHI S
### Register Number: 212223040144
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1 = nn.Linear(1,4)
        self.fc2 = nn.Linear(4,7)
        self.fc3 = nn.Linear(7,4)
        self.fc4 = nn.Linear(4,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.fc4(x)

    return x




# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
   X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')



```
## Dataset Information


![ஸ்கிரீன்ஷாட் 2025-03-03 113059](https://github.com/user-attachments/assets/80a5a47a-16a3-4061-ae45-05238647ebb4)

## OUTPUT

### Training Loss Vs Iteration Plot
![ஸ்கிரீன்ஷாட் 2025-03-03 114409](https://github.com/user-attachments/assets/aa23f750-199c-47bc-b957-c5cb7b32fff7)


### New Sample Data Prediction

![ஸ்கிரீன்ஷாட் 2025-03-03 114520](https://github.com/user-attachments/assets/57024be8-0ac0-4abb-b7ef-846147d1bd8d)


## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
