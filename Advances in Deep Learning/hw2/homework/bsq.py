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
        
        # Down projection to codebook_bits dimensions
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits, bias=False)
        
        # Up projection back to embedding_dim
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        # Down projection
        proj = self.down_proj(x)
        
        # L2 normalization
        norm = torch.nn.functional.normalize(proj, p=2, dim=-1)
        
        # Differentiable sign
        return diff_sign(norm)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        # Up projection
        return self.up_proj(x)

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
        # Use exponentiation instead of bit shifting for Apple Silicon compatibility
        return (x * (2 ** torch.arange(x.size(-1), device=x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        # Use exponentiation instead of bit shifting for Apple Silicon compatibility
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
        
        # Create the BSQ module for quantization
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=self.bottleneck)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to indices using the patch encoder and BSQ
        """
        # First encode with the patch encoder
        patches = super().encode(x)
        
        # Then encode to indices with BSQ
        return self.bsq.encode_index(patches)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode indices to an image using BSQ and the patch decoder
        """
        # First decode indices to patch embeddings with BSQ
        patches = self.bsq.decode_index(x)
        
        # Then decode to image with the patch decoder
        return super().decode(patches)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to binary codes using the patch encoder and BSQ
        """
        # First encode with the patch encoder
        patches = super().encode(x)
        
        # Then encode to binary codes with BSQ
        return self.bsq.encode(patches)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode binary codes to an image using BSQ and the patch decoder
        """
        # First decode binary codes to patch embeddings with BSQ
        patches = self.bsq.decode(x)
        
        # Then decode to image with the patch decoder
        return super().decode(patches)

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
        # Encode with both patch encoder and BSQ
        encoded = self.encode(x)
        
        # Decode with both BSQ and patch decoder
        decoded = self.decode(encoded)
        
        # Monitor codebook usage as suggested
        indices = self.encode_index(x)
        device = indices.device
        
        try:
            # Count the occurrences of each index
            cnt = torch.bincount(indices.flatten(), minlength=2**self.codebook_bits)
            
            # Calculate the statistics
            additional_losses = {
                "cb0": (cnt == 0).float().mean().detach(),  # Percentage of unused codes
                "cb2": (cnt <= 2).float().mean().detach(),  # Percentage of rarely used codes (≤ 2 occurrences)
                "cb10": (cnt <= 10).float().mean().detach(),  # Percentage of codes with ≤ 10 occurrences
                "cb_max": cnt.max().float().detach(),  # Maximum occurrences of any code
                "cb_entropy": (-torch.log2(cnt.float() / cnt.sum() + 1e-10) * (cnt.float() / cnt.sum())).sum().detach(),  # Entropy
            }
        except Exception as e:
            # In case of any issues (e.g., with Apple Silicon), just return basic stats
            additional_losses = {
                "error": torch.tensor(0.0, device=device)
            }
        
        return decoded, additional_losses
