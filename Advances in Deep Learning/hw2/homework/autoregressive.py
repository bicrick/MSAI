import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # Create token embedding
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Add positional embedding (optional but can help with performance)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, 600, d_latent))  # 600 is enough for 30x20 images
        
        # Create a causal mask for the transformer
        # This ensures autoregressive property - each position can only attend to previous positions
        
        # Create the transformer encoder (using it as a decoder with causal mask)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4*d_latent,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection layer
        self.output_projection = torch.nn.Linear(d_latent, n_tokens)

    def _create_causal_mask(self, seq_len: int, device=None) -> torch.Tensor:
        """Create a causal mask for the transformer"""
        return torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, h, w = x.shape
        
        # Flatten the 2D token grid into a 1D sequence
        x_flat = x.reshape(batch_size, -1)  # (B, h*w)
        seq_len = x_flat.size(1)
        
        # Embed the tokens
        x_embed = self.token_embedding(x_flat)  # (B, h*w, d_latent)
        
        # Add positional embedding
        x_embed = x_embed + self.pos_embedding[:, :seq_len, :]
        
        # For autoregressive prediction:
        # 1. Shift the sequence by prepending a special "start" token
        # We'll use zeros as our start token embedding
        start_embedding = torch.zeros(batch_size, 1, self.d_latent, device=x.device)
        x_shifted = torch.cat([start_embedding, x_embed[:, :-1, :]], dim=1)
        
        # Create causal attention mask
        mask = self._create_causal_mask(seq_len, device=x.device)
        
        # Pass through transformer with causal mask
        transformer_out = self.transformer(x_shifted, mask=mask)  # (B, h*w, d_latent)
        
        # Project to token probabilities
        token_logits = self.output_projection(transformer_out)  # (B, h*w, n_tokens)
        
        # Reshape back to 2D grid
        token_logits = token_logits.reshape(batch_size, h, w, self.n_tokens)
        
        # Return token logits and empty dict (no additional losses)
        return token_logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """Generate new token images autoregressively"""
        device = device or (next(self.parameters()).device)
        seq_len = h * w
        
        # Initialize with all zeros (placeholder for generation)
        generated_tokens = torch.zeros((B, seq_len), dtype=torch.long, device=device)
        
        # Generate tokens one by one autoregressively
        for i in range(seq_len):
            # Get the current sequence
            current_seq = generated_tokens[:, :i] if i > 0 else None
            
            if i == 0:
                # For the first token, use a zero embedding
                x_embed = torch.zeros(B, 1, self.d_latent, device=device)
                
                # Add positional embedding
                x_embed = x_embed + self.pos_embedding[:, 0:1, :]
                
                # No need for a mask with single token
                transformer_out = self.transformer(x_embed)
            else:
                # Embed all generated tokens so far
                x_embed = self.token_embedding(current_seq)  # (B, i, d_latent)
                
                # Add positional embedding
                x_embed = x_embed + self.pos_embedding[:, :i, :]
                
                # Create a causal mask for the current sequence length
                mask = self._create_causal_mask(i, device=device)
                
                # Get next token prediction
                transformer_out = self.transformer(x_embed, mask=mask)
            
            # Get the next token probabilities (last position in sequence)
            next_token_logits = self.output_projection(transformer_out[:, -1, :])  # (B, n_tokens)
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            
            # Add the new token to our generated sequence
            generated_tokens[:, i] = next_token
        
        # Reshape to 2D grid
        return generated_tokens.reshape(B, h, w)
