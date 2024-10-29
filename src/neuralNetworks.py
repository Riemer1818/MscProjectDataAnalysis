import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

class BaselineNN(BaseModel):
    def __init__(self, input_dim, output_dim):
        super(BaselineNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DropoutNN(BaseModel):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(DropoutNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SkipConnectionNN(BaseModel):
    def __init__(self, input_dim, output_dim):
        super(SkipConnectionNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.residual_projection = nn.Linear(input_dim, 128)

    def forward(self, x):
        residual = self.residual_projection(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x + residual  # Add skip connection here
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DropoutSkipConnectionNN(BaseModel):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(DropoutSkipConnectionNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.residual_projection = nn.Linear(input_dim, 128)

    def forward(self, x):
        residual = self.residual_projection(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = x + residual  # Add skip connection here
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
