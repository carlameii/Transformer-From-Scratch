from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, List

import re


@dataclass
class GPTConfig:
    # default test values -- too small for a real language model, but big enough for testing
    d_vocab: int = 10_000
    d_model: int = 128
    d_mlp: int = 512
    n_heads: int = 4
    d_head: int = 32
    n_layers: int = 6
    act_fn: type[nn.Module] = nn.ReLU

    @property
    def n_params(self) -> tuple[int]:
        "an estimate of the number of parameters"
        return (
            self.d_vocab * self.d_model  # embeddings (and tied unembeddings)
            + (
                    self.d_model * self.d_mlp * 2  # mlp weights
                    + self.d_model + self.d_mlp  # mlp bias
                    + self.n_heads * (  # number of heads
                            4 * self.d_model * self.d_head  # 4 because Q, K, O, V
                    )
            ) * self.n_layers,  # for each layer
        )


# note: the residual stream is `n_context` by `d_model`

# this is the row-wise (last dimension) softmax of x
# F.softmax(x, dim=-1)

class AttentionHead(nn.Module):
    def __init__(self, cfg: GPTConfig):
        print("Attention Head Constructor...")
        super().__init__()
        self.relu = nn.ReLU()
        self.d_vocab = cfg.d_vocab
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.wq = nn.Linear(self.d_model, self.d_head)
        self.wk = nn.Linear(self.d_model, self.d_head)
        self.wv = nn.Linear(self.d_model, self.d_head)
        self.wo = nn.Linear(self.d_head, self.d_model)

    def forward(self, x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        def masking_matrix(n_context):
            mask = torch.zeros((n_context, n_context))  # Start with all 0s
            mask[torch.triu(torch.ones((n_context, n_context)), diagonal=1) == 1] = -float(
                'inf')  # Set above diagonal to -inf
            return mask

        M = masking_matrix(x.shape[0])
        # softmax_argument = x*self.wq*torch.transpose(self.wk)*torch.transpose(x) + M
        wk_out = torch.transpose(self.wk(x), 0, 1)
        # print("WK shape ", wk_out.shape)
        wq_out = self.wq(x)
        # print("WQ shape ", wq_out.shape)
        softmax_out = F.softmax((wq_out @ wk_out + M), dim=-1)
        # print("Softmax shape ", softmax_out.shape)
        wv_out = self.wv(x)
        # print("WV shape ", wv_out.shape)
        wo_out = self.wo(wv_out)

        result = softmax_out @ wo_out
        # print("Final A Shape ", result.shape)
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        print("MultiHeadedAttention Constructor...")
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head

        # Multi-head attention
        self.attention_heads = nn.ModuleList([AttentionHead(cfg) for _ in range(self.n_heads)])

        # Linear projection to project summed outputs back to d_model
        self.wo = nn.Linear(self.d_model, self.d_model)  # Fix the output size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.attention_heads]
        # print("Head output shape: ", head_outputs[0].shape)

        summed_heads = torch.sum(torch.stack(head_outputs), dim=0)  # Sum over heads -> (n_context, d_head)

        summed_heads += x  # Element-wise addition (ensures same shape)

        # Project back to d_model
        output = self.wo(summed_heads)  # (n_context, d_model)

        return output


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        print("MLP Constructor...")
        super().__init__()

        # config
        self.d_model = cfg.d_model
        self.d_mlp = cfg.d_mlp

        # affine transformations
        self.lin1 = nn.Linear(self.d_model, self.d_mlp)
        # with nonlinearities in between
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.d_mlp, self.d_model)

    def forward(self, x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        # apply things in sequence
        out = self.lin1(x.flatten(start_dim=1))
        out = self.relu(out)
        out = self.lin2(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        print("TransformerBlock Constructor...")
        super().__init__()
        # uses `MultiHeadedAttention` and `MLP`
        self.multiheadattn = MultiHeadedAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: Float[torch.Tensor, "n_context d_vocab"]) -> Float[torch.Tensor, "n_context d_vocab"]:
        out = self.multiheadattn(x)
        out = self.mlp(out) + x
        return out


class Transformer(nn.Module):

    def __init__(self, cfg: GPTConfig):
        print("**" * 30)
        print("Transformer Constructor...")
        super().__init__()
        self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.unembedding = nn.Linear(cfg.d_model, cfg.d_vocab)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

    # uses `MultiHeadedAttention` and `MLP`
    # uses nn.Embedding for the embedding, transpose of it for the unembedding

    def forward(self, x: Int[torch.Tensor, "n_context"]) -> Float[torch.Tensor, "n_context d_vocab"]:
        out = self.embedding(x)
        # print(out.shape)
        for block in self.transformer_blocks:
            out = block(out)
        out = F.softmax(self.unembedding(out), dim=-1)
        return out


"""
Embedding:

input: list[int] \in Z^{n_c}
nn.Embedding: Z^{n_C} -> R^{n_c * d_m}

output:
nn.Linear(d_model, d_vocab): R^{n_c * d_m} -> R^{n_c * d_v}
"""


class TextProcessor:
    def __init__(self, text, cfg):
        print("==" * 30)
        print("TextFinder Constructor...")
        self.text = text
        self.word_index = self.create_word_index(text)
        self.n_context = cfg.n_context

    def create_word_index(self):
        # Create a word index mapping each word to a unique index
        words = re.findall(r'\b\w+\b', self.text.lower())
        sorted_words = sorted(set(words))
        word_to_index = {word: idx for idx, word in enumerate(sorted_words)}
        return word_to_index

    def text_to_tensor(self):
        # Convert the text into a tensor representation
        words = re.findall(r'\b\w+\b', self.text.lower())
        int_sequence = [self.word_index[word] for word in words if word in self.word_index]
        return torch.tensor(int_sequence, dtype=torch.long)

    def split_into_context_tensors(self):
        # for splitting long text into separate tensors of length n_context
        tensor = self.text_to_tensor()
        n = self.n_context
        trimmed_length = (len(tensor) // n) * n  # Ensure the length is a multiple of n_context
        tensor = tensor[:trimmed_length]  # Trim excess elements
        return [tensor[i:i + n] for i in range(0, len(tensor), n)]


class Trainer:
    def __init__(self, model: Transformer, text: str, optimizer: torch.optim.Optimizer,
                 device: torch.device = ("mps" if torch.mps.is_available() else "cpu"),
                 batch_size: int = 1, max_batches: Optional[int] = None, print_interval: int = 1,
                 epochs: int = 1):
        print("Trainer Constructor...")
        self.model = model
        self.text = text
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.print_interval = print_interval
        self.epochs = epochs
        self.dataset = TextProcessor(text)
        self.data_tensor = self.dataset.text_to_tensor().to(device)
        self.dataloader = self.create_dataloader()

        # Move model to device
        self.model.to(device)

    def create_dataloader(self):
        # Create batches
        data_batches = self.data_tensor.unfold(0, self.batch_size, 2)
        return DataLoader(data_batches, batch_size=1, shuffle=False)  # Using DataLoader to load batches

    def train(self):
        print(f"Training with device: {self.device}")
        training_records: List[dict] = []
        self.model.train()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Training"):
                batch = batch.squeeze(0)  # Remove extra dimension from the batch

                inputs = batch[:-1]
                targets = batch[1:]

                # forward pass
                logits = self.model(inputs)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record progress
                training_records.append({
                    "batch": i,
                    "loss": loss.item(),
                })

                if i % self.print_interval == 0:
                    print(f"Batch {i}, Loss: {loss.item()}")

                if self.max_batches is not None and i >= self.max_batches:
                    break

        return self.model, training_records


def main():
    gpt_config = GPTConfig()
    gpt_model = Transformer(gpt_config)

    optimizer = optim.Adam(gpt_model.parameters(), lr=0.001)

    some_text = """
        In reality, of course, we don't construct such chains explicitly, but instead we want them to learn from data.
        To put something in a markov chain or neural network, we need to turn it into numbers. this is straightforward for images: each pixel is already a number!
        In computers, text is stored as a sequence of numbers. Our neural network, in principle, can learn to predict the next number in the sequence. However, each number usually represents a single letter, or even just part of a letter.
    """

    trainer = Trainer(gpt_model, some_text, optimizer, epochs=1, batch_size=1, print_interval=1)

    print("Starting training...")
    trained_model, training_records = trainer.train()

    # Output the training records (losses)
    print("Training complete.")
    for record in training_records:
        print(f"Batch {record['batch']}, Loss: {record['loss']}")


if __name__ == "__main__":
    main()