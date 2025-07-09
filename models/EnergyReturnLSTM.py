import torch
import torch.nn as nn

class EnergyReturnLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(EnergyReturnLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Shared LSTM output goes to two heads
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Predict E_required_return (regression)
        )

        self.time_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Predict T_required_return (regression)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_len, hidden_size)
        final_hidden = lstm_out[:, -1, :]  # take last timestep's output

        energy_output = self.energy_head(final_hidden)
        time_output = self.time_head(final_hidden)

        return energy_output.squeeze(-1), time_output.squeeze(-1)
