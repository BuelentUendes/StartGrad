# Script for XAI classifier and Perturbation network
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StateClassifier(nn.Module):

    def __init__(
            self,
            feature_size,
            hidden_size,
            dropout_rate=0.5,
            n_state=1,  # Binary classification problem
            rnn_type="GRU",
            bidirectional=False,
    ):

        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.n_state = n_state
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self._init_model()

    def _init_model(self):

        # Initialize RNN layer
        rnn_layer = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        D = 2 if self.bidirectional else 1
        # the RNN layer expects then B x T x D which is the format of our data
        self.rnn = rnn_layer(self.feature_size, self.hidden_size, batch_first=True,
                             bidirectional=self.bidirectional)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=D * self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(D * self.hidden_size, self.n_state)
        )

    def forward(self, x):
        output, encoding = self.rnn(x)
        batch_size, seq_len, hidden_size = output.size()

        # We reshape the output to Batch size x sequence length, hidden_dimension
        output = self.fc(output.reshape(batch_size * seq_len, -1))

        if self.n_state == 1:
            # Binary cross entropy loss
            final_out = output.reshape(batch_size, -1)
        else:
            # Cross entropy loss expects it to be Batch x Classes x Dimension
            final_out = output.reshape(batch_size, seq_len, self.n_state).permute(0, 2, 1)

        return final_out


class StateClassifierMIMIC(nn.Module):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn_type: str = "LSTM",
        rnn_layers: int = 1,
        regres: bool = True,
        p_dropout: float = 0.5,
        bidirectional: bool = False,
        return_all: bool = False,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super(StateClassifierMIMIC, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.rnn_layers = rnn_layers
        self.regres = regres
        self.return_all = return_all

        # Choose RNN type with 2 layers
        rnn_class = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_class(feature_size, hidden_size, num_layers=rnn_layers, bidirectional=bidirectional, batch_first=True)

        # Regressor
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            # nn.ReLU(), # This was not before here! #t
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size, n_state)
        )

        self.to(self.device)

    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Initialize hidden (and cell) states for the RNN layers.
        """
        if isinstance(self.rnn, nn.LSTM):
            return (
                torch.zeros(self.rnn_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.rnn_layers, batch_size, self.hidden_size).to(self.device)
            )
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_size).to(self.device)

    def forward(self, x: torch.Tensor, past_state: Optional[Tuple[torch.Tensor, ...]] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        batch_size = x.size(0)

        if past_state is None:
            past_state = self.init_hidden_state(batch_size)

        # Forward pass through RNN layers
        rnn_out, _ = self.rnn(x, past_state)

        if self.regres:
            if not self.return_all:
                # Use the last time step for classification
                output = self.regressor(rnn_out[:, -1, :])
            else:
                # Use all time steps
                output = self.regressor(rnn_out.contiguous().view(-1, self.hidden_size))
                output = output.view(batch_size, -1, self.n_state).permute(0, 2, 1)
            return output
        else:
            return rnn_out[:, -1, :]


class PerturbationNetwork(nn.Module):

    def __init__(
            self,
            feature_size,
            hidden_size,
            rnn_type="GRU",
            bidirectional=True,
            signal_length=200,
            normalize=True,
    ):

        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.signal_length = signal_length
        self.normalize = normalize
        self._init_model()

    def _init_model(self):
        # Initialize RNN layer
        rnn_layer = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        D = 2 if self.bidirectional else 1
        # the RNN layer expects then B x T x D which is the format of our data
        self.rnn = rnn_layer(self.feature_size, self.hidden_size, batch_first=True,
                             bidirectional=self.bidirectional)

        self.fc = nn.Sequential(
            nn.Linear(D * self.hidden_size, self.feature_size)
        )

    def forward(self, x):

        output, _ = self.rnn(x)
        # We follow the implementation: normalization is also used
        # https://github.com/anonymous8293/factai/blob/main/tint/models/rnn.py
        output = F.normalize(output, dim=-1, p=2) if self.normalize else output
        output = self.fc(output)
        return output



