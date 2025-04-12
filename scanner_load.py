import torch
from transformers import T5Tokenizer
import torch.nn as nn


class IntelGuardNet(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, tokenizer):
        super(IntelGuardNet, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Base embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Syntax-aware components
        self.token_type_embeddings = nn.Embedding(4, d_model)  # 0=command, 1=flag, 2=value, 3=special
        self.position_embeddings = nn.Embedding(512, d_model)  # Increased position awareness

        # Enhanced transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )

        # Output layers with syntax gate
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.syntax_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def _get_token_types(self, input_ids):
        """Determine token types (command, flag, value) for syntax awareness"""
        batch_size, seq_len = input_ids.shape
        token_types = torch.zeros_like(input_ids)

        for b in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
            for i, token in enumerate(tokens):
                if token == self.tokenizer.pad_token:
                    token_types[b, i] = 3  # Special
                elif i == 0 or (i == 1 and tokens[0] == self.tokenizer.bos_token):
                    token_types[b, i] = 0  # Command
                elif token.startswith('-'):
                    token_types[b, i] = 1  # Flag
                elif token.replace('.', '').isdigit():
                    token_types[b, i] = 2  # Value
                else:
                    token_types[b, i] = 3  # Other
        return token_types

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        # Get base embeddings
        src = self.embedding(input_ids)
        tgt = self.embedding(decoder_input_ids)

        # Add syntax information
        src_token_types = self._get_token_types(input_ids)
        tgt_token_types = self._get_token_types(decoder_input_ids)

        src = src + self.token_type_embeddings(src_token_types)
        tgt = tgt + self.token_type_embeddings(tgt_token_types)

        # Add position information
        src_positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        tgt_positions = torch.arange(0, decoder_input_ids.size(1), device=decoder_input_ids.device).unsqueeze(0)

        src = src + self.position_embeddings(src_positions)
        tgt = tgt + self.position_embeddings(tgt_positions)

        # Transformer processing
        output = self.transformer(src, tgt)

        # Syntax-gated output
        gate = self.syntax_gate(output)
        output = output * gate  # Emphasize syntax-correct features

        return self.fc_out(output)

    def generate(self, input_ids, max_length=50):
        """Syntax-aware generation method"""
        self.eval()
        device = input_ids.device

        # Initialize with start token
        decoder_input = torch.tensor([[self.tokenizer.bos_token_id]], device=device)

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input
                )

            # Get next token probabilities
            next_token_logits = outputs[:, -1, :]

            # Debug: print logits
            print(f"Next Token Logits: {next_token_logits}")

            current_text = self.tokenizer.decode(decoder_input[0], skip_special_tokens=True)
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            # Debug: print token probabilities
            print(f"Next Token Probabilities: {next_token_probs}")

            # Enforce syntax rules
            if not current_text.strip():  # Must start with command
                next_token_probs[:, :self.tokenizer.vocab_size] = -float('inf')
                next_token_probs[:, self.tokenizer.convert_tokens_to_ids("nmap")] = 1.0
            elif current_text.endswith(' -'):  # Must be followed by flag character
                valid_flags = [i for t, i in self.tokenizer.vocab.items()
                               if t.replace('▁', '').isalpha() and len(t.replace('▁', '')) == 1]
                next_token_probs[:, :self.tokenizer.vocab_size] = -float('inf')
                next_token_probs[:, valid_flags] = 1.0 / len(valid_flags)

            # Debug: print selected next token
            next_token = torch.argmax(next_token_probs, dim=-1)
            print(f"Selected Next Token: {next_token}")

            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Debug: print the final generated command
        generated_command = self.tokenizer.decode(decoder_input[0], skip_special_tokens=True)
        print(f"Generated Command: {generated_command}")
        return generated_command


def generate_command(target, iden, stat, ports, os):
    input_text = f"Target: {target}, Iden: {iden}, Stat: {stat}, Ports: {ports}, OS: {os}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Debug: print tokenized input
    print(f"Tokenized Input IDs: {inputs['input_ids']}")

    decoder_input = torch.tensor([[tokenizer.pad_token_id]]).to(device)
    for _ in range(50):  # max length
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input
        )

        # Debug: print outputs
        print(f"Model Output: {outputs}")

        next_token = torch.argmax(outputs[:, -1, :], dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=-1)

        # Debug: print next token and decoder input
        print(f"Next Token: {next_token}")
        print(f"Updated Decoder Input: {decoder_input}")

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(decoder_input[0], skip_special_tokens=True)


tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("complete_model.pth", weights_only=False, map_location=device)

# Call the function with debug
cmd = generate_command("127.0.0.1", "ID-1", "up", "'8080': 'Apache Tomcat 9.0.12'", "Linux")
print(f"Final Command: {cmd}")
