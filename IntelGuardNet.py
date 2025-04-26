import torch
import torch.nn as nn


class IntelGuardNet(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, tokenizer):
        super(IntelGuardNet, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Base embeddings with better initialization
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)

        # Syntax-aware components
        self.token_type_embeddings = nn.Embedding(4, d_model)  # 0=command, 1=flag, 2=value, 3=special
        self.position_embeddings = nn.Embedding(512, d_model)  # Increased position awareness

        # Enhanced transformer with layer normalization
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output layers
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.syntax_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Initialize weights properly
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

        # Transformer processing with layer norm
        output = self.transformer(src, tgt)
        output = self.norm(output)

        # Syntax-gated output
        gate = self.syntax_gate(output)
        output = output * gate

        return self.fc_out(output)

    def generate(self, input_ids, max_length=50):
        """Syntax-aware generation method"""
        self.eval()
        device = input_ids.device
        # Use pad_token_id or simply 0 as the initial token
        start_token_id = self.tokenizer.pad_token_id or 0

        decoder_input = torch.tensor([[start_token_id]], device=device)

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input
                )

            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return decoder_input
