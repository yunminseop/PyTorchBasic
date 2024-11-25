import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비
# 2. 모델 정의
# 3. 손실 함수 및 옵티마이저 설정
# 4. 학습 루프
# 5. 학습 결과 확인


torch.manual_seed(42)
x = torch.randn(100, 1) # 입력 데이터
print("x:", x)
print("type of x:", type(x) )

y = 2 * x + 3 + 0.1 * torch.randn(100, 1) # 출력 데이터

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)
        #### nn.Linear(1, 1)이 뭘까
        self.prompt = nn.Linear(1, 2)
        self.final = nn.Linear(2,1)

    def forward(self, x):
        x = self.linear(x)
        #print(f"self.linear(x): {x}")
        # print(f"x: {x.shape}")
        prompt = self.prompt(x)
        #print(f"self.prompt(x): {prompt}")
        # print(f"prompt: {prompt.shape}")
        x1 = self.fc1(x) + x
        #print(f"self.fc1(x) + x: {x1}")
        x2 = self.fc2(x1)
        # print(f"x1: {x1.shape}")
        x = self.fc2(x1) + prompt
        #print(f"self.fc2(x1) + prompt: {x}")
        # print(f"x: {x.shape}")
        x = self.relu(x) + x2
        
        return self.final(x)
    

model = LinearRegressionModel()

loss_function = nn.MSELoss()
optimizer = optim.SGD(params = model.linear.parameters(), lr=1.2)
# lr = learning rate (학습률)

num_epochs = 100

for epoch in range(num_epochs):
    y_pred = model(x)
    loss = loss_function(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1}번째 Epoch_ Loss: {loss.item():.4f}")
