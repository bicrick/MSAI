from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # CNN Architecture for classification
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x32x32
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x16x16
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x8x8
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256x4x4
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Adding dropout for regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Forward pass through the network
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        logits = self.classifier(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Number of filters in each layer
        filters = [16, 32, 64, 128, 256]
        
        # Encoder (downsampling) blocks
        # 1st encoder block
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd encoder block
        self.enc2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd encoder block
        self.enc3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4th encoder block
        self.enc4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[4], filters[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling) blocks with skip connections
        # 1st decoder block
        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(filters[3] * 2, filters[3], kernel_size=3, padding=1),  # *2 because of skip connection
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )
        
        # 2nd decoder block
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(filters[2] * 2, filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )
        
        # 3rd decoder block
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(filters[1] * 2, filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )
        
        # 4th decoder block
        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutional layers for segmentation and depth
        self.seg_output = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.depth_output = nn.Sequential(
            nn.Conv2d(filters[0], 1, kernel_size=1),
            nn.Sigmoid()  # Ensures depth values are in range [0, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder path with skip connections
        enc1_out = self.enc1(z)
        z = self.pool1(enc1_out)
        
        enc2_out = self.enc2(z)
        z = self.pool2(enc2_out)
        
        enc3_out = self.enc3(z)
        z = self.pool3(enc3_out)
        
        enc4_out = self.enc4(z)
        z = self.pool4(enc4_out)
        
        # Bridge
        z = self.bridge(z)
        
        # Decoder path with skip connections
        z = self.up1(z)
        z = torch.cat([z, enc4_out], dim=1)  # Skip connection
        z = self.dec1(z)
        
        z = self.up2(z)
        z = torch.cat([z, enc3_out], dim=1)  # Skip connection
        z = self.dec2(z)
        
        z = self.up3(z)
        z = torch.cat([z, enc2_out], dim=1)  # Skip connection
        z = self.dec3(z)
        
        z = self.up4(z)
        z = torch.cat([z, enc1_out], dim=1)  # Skip connection
        z = self.dec4(z)
        
        # Generate outputs
        logits = self.seg_output(z)
        depth = self.depth_output(z).squeeze(1)  # Convert from (b, 1, h, w) to (b, h, w)
        
        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        
        # Raw depth is already normalized by the sigmoid in depth_output
        depth = raw_depth
        
        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


def debug_detector(batch_size: int = 16):
    """
    Test the detector's prediction speed
    """
    import time
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create random test data
    sample_batch = torch.rand(batch_size, 3, 96, 128).to(device)
    print(f"Input shape: {sample_batch.shape}")

    # Load model
    model = load_model("detector", in_channels=3, num_classes=3, with_weights=True).to(device)
    model.eval()
    
    # Warm-up run
    with torch.inference_mode():
        model.predict(sample_batch)
    
    # Time multiple predictions
    num_runs = 10
    total_time = 0
    
    with torch.inference_mode():
        for i in range(num_runs):
            start_time = time.time()
            pred, depth = model.predict(sample_batch)
            end_time = time.time()
            run_time = end_time - start_time
            total_time += run_time
            print(f"Run {i+1}: {run_time:.4f}s")
    
    print(f"Average prediction time for batch of {batch_size}: {total_time/num_runs:.4f}s")
    print(f"Prediction output shapes: {pred.shape}, {depth.shape}")


if __name__ == "__main__":
    # debug_model()  # Test classifier
    debug_detector()  # Test detector speed
