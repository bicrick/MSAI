import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim
        
        # Linear projections for encoding and decoding
        self.proj_down = torch.nn.Linear(embedding_dim, codebook_bits, bias=False)
        self.proj_up = torch.nn.Linear(codebook_bits, embedding_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        # Project down to codebook_bits dimensions
        x = self.proj_down(x)
        
        # L2 normalization (normalize along the last dimension)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        # Apply differentiable sign
        return diff_sign(x_norm)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.proj_up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (2 ** torch.arange(x.size(-1), device=x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits, device=x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        
        # Initialize the BSQ module
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # First use the encoder from PatchAutoEncoder to get latent representation
        z = super().encode(x)
        # Then use BSQ to encode to indices
        return self.bsq.encode_index(z)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # First use BSQ to decode indices to latent representation
        z = self.bsq.decode_index(x)
        # Then use the decoder from PatchAutoEncoder to reconstruct the image
        return super().decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # First encode with parent encoder
        z = super().encode(x)
        # Then apply BSQ encoding
        return self.bsq.encode(z)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Decode with BSQ first
        z = self.bsq.decode(x)
        # Then decode with parent decoder
        return super().decode(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        # Get encoded representation
        z = super().encode(x)
        
        # Apply BSQ
        z_bsq = self.bsq.encode(z)
        
        # Decode
        decoded_z = self.bsq.decode(z_bsq)
        x_hat = super().decode(decoded_z)
        
        # Additional tracking metrics
        with torch.no_grad():
            # Get indices
            indices = self.bsq.encode_index(z)
            
            # Calculate codebook usage statistics
            cnt = torch.bincount(indices.flatten(), minlength=2**self.codebook_bits)
            
            # Track unused and rarely used tokens
            stats = {
                "cb0": (cnt == 0).float().mean().detach(),  # Percentage of unused tokens
                "cb2": (cnt <= 2).float().mean().detach(),  # Percentage of tokens used ≤ 2 times
                "cb10": (cnt <= 10).float().mean().detach(), # Percentage of tokens used ≤ 10 times
                "diversity": (cnt > 0).float().sum().detach() / (2**self.codebook_bits), # Token diversity
            }
        
        return x_hat, stats
