
from dataclasses import dataclass
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
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

    def forward(self,
                x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        def masking_matrix(n_context):
            mask = torch.zeros((n_context, n_context))  # Start with all 0s
            mask[torch.triu(torch.ones((n_context, n_context)), diagonal=1) == 1] = -float('inf')  # Set above diagonal to -inf
            return mask

        M = masking_matrix(x.shape[0])
        # softmax_argument = x*self.wq*torch.transpose(self.wk)*torch.transpose(x) + M
        # wk_out = torch.transpose(self.wk(x), 0, 1)
        wk_out = self.wk(x).transpose(-2, -1)  # Correct transposition

        # wk_out = self.wk(x).transpose(-2, -1)
        # print("WK shape ", wk_out.shape)
        # print("WK shape: torch.transpose(self.wk(x), 0, 1) ", torch.transpose(self.wk(x), 0, 1).shape)
        # print("WK shape: self.wk(x).transpose(-2, -1) ", self.wk(x).transpose(-2, -1).shape)

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

    def forward(self,
                x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        # apply things in sequence
        # out = self.lin1(x.flatten(start_dim=1))
        out = self.lin1(x)  # No need to flatten
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

    # def forward(self,
    #             x: Float[torch.Tensor, "n_context d_vocab"]) -> Float[torch.Tensor, "n_context d_vocab"]:
    #     out = self.multiheadattn(x)
    #     out = self.mlp(out) + x
    #     return out

    # d_model instead of d_vocab
    def forward(self, x: Float[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        out = self.multiheadattn(x)  # Ensures the shape is (n_context, d_model)
        out = self.mlp(out) + x  # Residual connection
        return out


class Transformer(nn.Module):

    def __init__(self, cfg: GPTConfig):
        print("**"*30)
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

class TextFinder:
    def __init__(self, text):
        print("=="*30)
        print("TextFinder Constructor...")
        self.text = text
        self.word_index = self.create_word_index(text)


        # added below
        self.index_to_word = {idx: word for word, idx in self.word_index.items()}  # Reverse mapping

    # def create_word_index(self, text):
    #     # Create a word index mapping each word to a unique index
    #     words = re.findall(r'\b\w+\b', text.lower())
    #     sorted_words = sorted(set(words))
    #     # word_to_index = {word: idx for idx, word in enumerate(sorted_words)}
    #     # return word_to_index
    #     return {word: idx for idx, word in enumerate(sorted_words)}
    #
    # def text_to_tensor(self):
    #     # Convert the text into a tensor representation
    #     words = re.findall(r'\b\w+\b', self.text.lower())
    #     int_sequence = [self.word_index[word] for word in words if word in self.word_index]
    #     return torch.tensor(int_sequence, dtype=torch.long)

    def create_word_index(self, text):
        # Create a word index mapping each word to a unique index, with [UNK] token
        words = re.findall(r'\b\w+\b', text.lower())
        sorted_words = sorted(set(words))
        sorted_words.append("[UNK]")  # Add an UNK token at the end
        return {word: idx for idx, word in enumerate(sorted_words)}

    def text_to_tensor(self):
        # Convert the text into a tensor representation, with [UNK] handling
        words = re.findall(r'\b\w+\b', self.text.lower())
        int_sequence = [self.word_index.get(word, self.word_index["[UNK]"]) for word in words]
        return torch.tensor(int_sequence, dtype=torch.long)

    def text_to_tensor_for_prompt(self, prompt):
        # Convert the prompt into a tensor representation (based on how the words appear in self.dataset)
        words = re.findall(r'\b\w+\b', prompt.lower())
        int_sequence = [self.word_index.get(word, self.word_index["[UNK]"]) for word in words]
        return torch.tensor(int_sequence, dtype=torch.long)

    def tensor_to_text(self, tensor):
        # Convert the tensor back to words using the index_to_word mapping
        word_list = [self.index_to_word.get(idx.item(), "[UNK]") for idx in tensor]
        return " ".join(word_list)


class Trainer:
    """
    for mac M1 chip, use mps instead of cuda
    """
    def __init__(self, model: Transformer,
                 text: str, optimizer: torch.optim.Optimizer,
                 device: torch.device = ("cuda" if torch.cuda.is_available() else "cpu"),
                 sample_size: int = 1, max_samples: Optional[int] = None,
                 print_interval: int = 1,
                 epochs: int = 1):
        print("Trainer Constructor...")
        self.model = model
        self.text = text
        self.optimizer = optimizer
        self.device = device
        self.sample_size = sample_size
        self.max_samples = max_samples
        self.print_interval = print_interval
        self.epochs = epochs
        self.dataset = TextFinder(text)
        self.data_tensor = self.dataset.text_to_tensor().to(device)
        self.dataloader = self.create_dataloader()

        # Move model to device
        self.model.to(device)

    def create_dataloader(self):
        #data_samples = [self.data_tensor[i:i+self.sample_size] for i in range(0, len(self.data_tensor)-self.sample_size)]
        #return DataLoader(data_samples, batch_size=1, shuffle=False)
        # Create batches
        # TODO double check the unfold
        data_samples = self.data_tensor.unfold(0, self.sample_size, self.sample_size)
        for data_sample in data_samples:
            print("Data sample: ", data_sample)
        return DataLoader(data_samples, batch_size=1, shuffle=False)  # Using DataLoader to load batches

    def train(self):
        print(f"Training with device: {self.device}")
        training_records: List[dict] = []
        self.model.train()
        loss_values = []
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for i, sample in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Training"):
                sample = sample.squeeze(0)  # Remove extra dimension from the sample

                inputs = sample[:-1]
                targets = sample[1:]

                # forward pass
                probabilities = self.model(inputs)
                log_probabilities = torch.log(probabilities)

                # Calculate loss using NLLLoss
                loss = F.nll_loss(log_probabilities.view(-1, log_probabilities.size(-1)), targets.view(-1))
                
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record progress
                training_records.append({
                    "sample": i,
                    "loss": loss.item(),
                })
                loss_values.append(loss.item())  # Store loss value


                if i % self.print_interval == 0:
                    print(f"Sample {i}, Loss: {loss.item()}")

                if self.max_samples is not None and i >= self.max_samples:
                    break
                
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label="Training Loss")
        plt.xlabel("Sample")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        return self.model, training_records


    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0) -> str:
        self.model.eval()

        text_finder = self.dataset
        input_tensor = text_finder.text_to_tensor_for_prompt(prompt).unsqueeze(0).to(self.device)

        generated_tokens = input_tensor.squeeze(0).tolist()

        for _ in range(max_tokens):
            logits = self.model(input_tensor)
            logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()

            generated_tokens.append(next_token)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

        # Use `index_to_word` instead of `word_index`
        generated_text = self.dataset.tensor_to_text(torch.tensor(generated_tokens))

        return generated_text



def get_gutenberg_book(
	id: int|None = 84,
	data_temp: Path|str = "../data/gutenberg_data",
	remove_gutenberg_meta: bool = True,
) -> str:
	
	data_temp = Path(data_temp)
	data_temp.mkdir(parents=True, exist_ok=True)
	
	url: str = f"https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
	data_path: Path = Path(data_temp) / f"{id}.txt"
	data: str
	# read from cache if it exists
	if data_path.exists():
		with open(data_path, 'r', encoding='utf-8') as file:
			data = file.read()
	else:
		# download if it doesn't exist
		response = requests.get(url)
		response.raise_for_status()  # Ensure that the download was successful
		data = response.text

		# save to cache
		with open(data_path, 'w', encoding='utf-8') as file:
			file.write(data)

	# remove header/footer
	if remove_gutenberg_meta:
		data = '***'.join(data.split('***')[2:])
		data = '***'.join(data.split('***')[:-1])
	
	return data


def main():
    gpt_config = GPTConfig()
    gpt_model = Transformer(gpt_config)

    optimizer = optim.Adam(gpt_model.parameters(), lr=1e-4)

    some_text = """
        In reality, of course, we don't construct such chains explicitly, but instead we want them to learn from data.
        To put something in a markov chain or neural network, we need to turn it into numbers. this is straightforward for images: each pixel is already a number!
        In computers, text is stored as a sequence of numbers. Our neural network, in principle, can learn to predict the next number in the sequence. However, each number usually represents a single letter, or even just part of a letter.
    """
    some_book = get_gutenberg_book(data_temp="./gutenberg_data")
    
    trainer = Trainer(gpt_model, some_book, optimizer, epochs=1, sample_size=50, print_interval=100)

    print("Starting training...")
    trained_model, training_records = trainer.train()

    # Output the training records (losses)
    print("Training complete.")
    for record in training_records:
        print(f"Sample {record['sample']}, Loss: {record['loss']}")

    print("**"*50)
    torch.save(trained_model, "model.pt")

    # Generate text with the trained model
    prompt = "Today I plan to complete the following tasks, "
    generated_text = trainer.generate(prompt)

    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()