"""LSTM sequence baseline for stock signal classification.

Motivation
----------
A marker observed that "the choice of RF, even with past data, remains
insufficiently justified. Time series is a specific type of data." The
criticism is well founded on two counts:

1. Random Forest treats each row as an independent observation. Financial
   series are autocorrelated and non-stationary, so that assumption is
   violated. The leakage experiment (exp01) showed the violation was not
   harmless — it inflated weighted F1 from 0.36 to 0.83.

2. Tree ensembles see only the current row's feature values. They cannot
   represent the *order* of recent observations, yet order is what
   distinguishes a rising market from a falling one at the same price level.

An LSTM (Hochreiter & Schmidhuber, 1997) addresses the second point directly:
it consumes a window of consecutive days and maintains hidden state across
them, so temporal patterns are representable in principle.

Including this baseline lets the model choice be settled by evidence rather
than assertion. If the LSTM also fails, that is itself informative — it
suggests the ceiling is imposed by the feature set and the difficulty of the
task, not by the choice of classifier.

Implementation notes
--------------------
Deliberately small: one LSTM layer, modest hidden size, early stopping on a
chronological validation split. The dataset is roughly 11,000 rows, so a large
network would overfit immediately. Training runs on CPU in a few minutes.

References:
    Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory',
        Neural Computation 9(8), pp. 1735-1780.
    Fischer, T. and Krauss, C. (2018) 'Deep learning with long short-term
        memory networks for financial market predictions', European Journal of
        Operational Research 270(2), pp. 654-669.
    Siami-Namini, S., Tavakoli, N. and Siami Namin, A. (2019) 'The performance
        of LSTM and BiLSTM in forecasting time series', IEEE International
        Conference on Big Data, pp. 3285-3292.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

#: Days of history fed to the network for each prediction.
DEFAULT_SEQUENCE_LENGTH = 20


def build_sequences(features: np.ndarray, labels: Sequence,
                    groups: Optional[Sequence] = None,
                    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-row features into overlapping fixed-length sequences.

    Each output sample is the `sequence_length` consecutive rows ending at
    position i, labelled with row i's target. When `groups` is supplied (ticker
    symbols here), sequences never span a group boundary — otherwise the tail
    of one stock's history would be concatenated with the head of another's.

    Args:
        features: Array of shape (n_rows, n_features), ordered by date within
            each group.
        labels: Target label per row.
        groups: Group identifier per row, typically the ticker. If omitted all
            rows are treated as one contiguous series.
        sequence_length: Number of consecutive rows per sample.

    Returns:
        Tuple of (X, y, row_index) where X has shape
        (n_samples, sequence_length, n_features), y holds the label of each
        sequence's final row, and row_index maps samples back to source rows.

    Raises:
        ValueError: If inputs disagree in length, sequence_length < 1, or no
            group is long enough to form a single sequence.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)

    if len(features) != len(labels):
        raise ValueError(
            f"features and labels differ in length: {len(features)} vs {len(labels)}")
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    if groups is None:
        groups = np.zeros(len(features), dtype=int)
    else:
        groups = np.asarray(groups)
        if len(groups) != len(features):
            raise ValueError(
                f"groups length {len(groups)} does not match features {len(features)}")

    sequences, targets, indices = [], [], []
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) < sequence_length:
            continue
        for end in range(sequence_length - 1, len(positions)):
            window = positions[end - sequence_length + 1:end + 1]
            sequences.append(features[window])
            targets.append(labels[positions[end]])
            indices.append(positions[end])

    if not sequences:
        raise ValueError(
            f"No group contains at least {sequence_length} rows; "
            "cannot build any sequence")

    return (np.stack(sequences), np.asarray(targets), np.asarray(indices))


@dataclass
class TrainingHistory:
    """Per-epoch training diagnostics."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


class LSTMClassifier:
    """Small LSTM classifier with a scikit-learn style interface.

    Exposes fit/predict so it can be dropped into the same comparison and
    statistical tests as the scikit-learn models, keeping the evaluation
    procedure identical across all candidates.

    Attributes:
        sequence_length: Days of history per sample.
        hidden_size: LSTM hidden state width.
        num_layers: Stacked LSTM layers.
        dropout: Dropout applied between layers and before the output head.
        classes_: Sorted class labels, populated by fit().
    """

    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                 hidden_size: int = 48, num_layers: int = 1,
                 dropout: float = 0.2, learning_rate: float = 1e-3,
                 batch_size: int = 128, max_epochs: int = 60,
                 patience: int = 8, random_state: int = 42):
        """Configure the network. Nothing is built until fit() is called."""
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.model = None
        self.history = TrainingHistory()

    def _build_network(self, n_features: int, n_classes: int):
        """Construct the torch module. Imported lazily so torch stays optional."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        class Net(nn.Module):
            def __init__(self, n_features, hidden, layers, dropout, n_classes):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden, n_classes)

            def forward(self, x):
                output, _ = self.lstm(x)
                # Classify from the final timestep's hidden state.
                return self.head(self.dropout(output[:, -1, :]))

        return Net(n_features, self.hidden_size, self.num_layers,
                   self.dropout, n_classes)

    def fit(self, X: np.ndarray, y: Sequence,
            groups: Optional[Sequence] = None,
            validation_fraction: float = 0.15,
            verbose: bool = False) -> "LSTMClassifier":
        """Train the network with early stopping on a chronological holdout.

        The validation split is the final `validation_fraction` of sequences in
        time order, never a random sample, so model selection cannot peek at
        future data. Class weights counteract the imbalance in Buy/Hold/Avoid.

        Args:
            X: Per-row feature array of shape (n_rows, n_features).
            y: Per-row labels.
            groups: Per-row group identifier (ticker) to prevent sequences
                spanning stocks.
            validation_fraction: Trailing fraction of sequences held out.
            verbose: Print per-epoch progress.

        Returns:
            self.

        Raises:
            ImportError: If PyTorch is unavailable.
            ValueError: If validation_fraction is outside (0, 1).
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for LSTMClassifier. "
                "Install with: pip install torch"
            ) from exc

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {validation_fraction}")

        sequences, targets, _ = build_sequences(
            X, y, groups, self.sequence_length)

        self.classes_ = np.unique(targets)
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        encoded = np.array([class_to_index[t] for t in targets])

        # Chronological validation split.
        split = int(len(sequences) * (1 - validation_fraction))
        split = max(1, min(split, len(sequences) - 1))

        X_train = torch.from_numpy(sequences[:split])
        y_train = torch.from_numpy(encoded[:split]).long()
        X_val = torch.from_numpy(sequences[split:])
        y_val = torch.from_numpy(encoded[split:]).long()

        self.model = self._build_network(sequences.shape[2], len(self.classes_))

        # Inverse-frequency class weights for the imbalanced target.
        counts = np.bincount(encoded[:split], minlength=len(self.classes_))
        weights = torch.tensor(
            len(encoded[:split]) / (len(self.classes_) * np.maximum(counts, 1)),
            dtype=torch.float32)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimiser = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            permutation = torch.randperm(len(X_train))
            epoch_loss = 0.0

            for start in range(0, len(X_train), self.batch_size):
                batch = permutation[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.model(X_train[batch]), y_train[batch])
                loss.backward()
                # Clip gradients: recurrent nets are prone to exploding gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                epoch_loss += loss.item() * len(batch)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = criterion(val_logits, y_val).item()
                val_accuracy = (val_logits.argmax(1) == y_val).float().mean().item()

            self.history.train_loss.append(epoch_loss / len(X_train))
            self.history.val_loss.append(val_loss)
            self.history.val_accuracy.append(val_accuracy)

            if verbose:
                print(f"    epoch {epoch + 1:3d}  train={epoch_loss / len(X_train):.4f}  "
                      f"val={val_loss:.4f}  val_acc={val_accuracy:.4f}")

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.history.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    self.history.stopped_early = True
                    if verbose:
                        print(f"    early stop at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray,
                groups: Optional[Sequence] = None) -> np.ndarray:
        """Predict labels for the rows that can terminate a full sequence.

        Rows in the first `sequence_length - 1` positions of each group have
        insufficient history and are therefore not predicted. Use
        `predict_with_index` when the caller needs to align predictions to rows.

        Args:
            X: Per-row feature array.
            groups: Per-row group identifier.

        Returns:
            Predicted labels, one per constructable sequence.

        Raises:
            RuntimeError: If called before fit().
        """
        return self.predict_with_index(X, groups)[0]

    def predict_with_index(self, X: np.ndarray, groups: Optional[Sequence] = None
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels and return the source row index of each prediction.

        Args:
            X: Per-row feature array.
            groups: Per-row group identifier.

        Returns:
            Tuple of (predicted labels, row indices they correspond to).

        Raises:
            RuntimeError: If called before fit().
        """
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Call fit() before predict()")

        import torch

        # Labels are unused for construction; pass a placeholder of right length.
        placeholder = np.zeros(len(X))
        sequences, _, indices = build_sequences(
            X, placeholder, groups, self.sequence_length)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(sequences))
            predicted = logits.argmax(1).numpy()

        return self.classes_[predicted], indices
