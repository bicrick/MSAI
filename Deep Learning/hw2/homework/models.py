"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        
        # Calculate input dimension: 3 channels * height * width
        input_dim = 3 * h * w
        
        # Single linear layer to project flattened input to class logits
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input: (b, 3, H, W) -> (b, 3*H*W)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Apply linear layer to get logits
        logits = self.classifier(x_flat)
        
        return logits


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
            hidden_dim: int, size of hidden layer
        """
        super().__init__()
        
        # Calculate input dimension: 3 channels * height * width
        input_dim = 3 * h * w
        
        # Create a sequential model with a hidden layer and ReLU activation
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input: (b, 3, H, W) -> (b, 3*H*W)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Apply MLP layers
        logits = self.layers(x_flat)
        
        return logits


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers (minimum 4)
        """
        super().__init__()
        
        # Calculate input dimension
        input_dim = 3 * h * w
        
        # Create layers list starting with input layer
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]
        
        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already have input and will add output
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        
        # Add output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        # Create sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Apply deep MLP layers
        logits = self.layers(x_flat)
        
        return logits


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers (minimum 4)
        """
        super().__init__()
        
        # Calculate input dimension
        input_dim = 3 * h * w
        
        # Input projection layer with double ReLU
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Create ModuleList for simple residual blocks
        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply residual layers
        for layer in self.residual_layers:
            # Compute residual and add to input (residual connection)
            x = x + layer(x)
        
        # Final projection to class logits
        logits = self.output_proj(x)
        
        return logits


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
