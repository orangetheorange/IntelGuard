import torch
from transformers import T5Tokenizer
from IntelGuardNet import IntelGuardNet

from pathlib import Path
import os

script_path = Path(__file__).resolve()

parent_dir = script_path.parent

os.chdir(parent_dir)

# 1. Load the tokenizer
print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
tokenizer.add_special_tokens({'additional_special_tokens': ['[CMD]', '[FLAG]', '[VAL]']})

# 2.1 recreate your model skeleton exactly as you did at training time
print("building model architecture…")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntelGuardNet(
    vocab_size=tokenizer.vocab_size + len(tokenizer.additional_special_tokens),
    d_model=256,
    nhead=4,
    num_layers=4,
    dim_feedforward=512,
    tokenizer=tokenizer
).to(device)

# 2.2 load only the weights you saved (final_model.pth is just a state_dict)
print("loading weights from final_model.pth…")
state_dict = torch.load(
    r"final_model.pth",
    map_location=device
)
model.load_state_dict(state_dict)
model.eval()


# 3. Prediction function
def predict_command(scan_line):
    model.eval()

    # Debug: Print input scan line
    print(f"Input scan line: {scan_line}")

    # Tokenize the input scan line
    inputs = tokenizer(
        scan_line,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Debug: Print tokenized input
    print(f"Tokenized inputs: {inputs}")

    # Start decoder input with a pad or any dummy token like "0"
    decoder_input_ids = torch.zeros(
        (inputs["input_ids"].size(0), 1),
        dtype=torch.long,
        device=device
    )

    # Debug: Print decoder input
    print(f"Decoder input ids: {decoder_input_ids}")

    with torch.no_grad():
        # Run the model
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids
        )

        # Debug: Print raw model output
        print(f"Raw model output: {outputs}")

    # Extract predictions
    preds = torch.argmax(outputs, dim=-1)

    # Debug: Print predictions
    print(f"Predicted token IDs: {preds}")

    # Decode predicted tokens
    predicted_command = tokenizer.decode(preds[0], skip_special_tokens=True)

    # Debug: Print the final predicted command
    print(f"Predicted command: {predicted_command}")

    return predicted_command


# 4. Main application loop
if __name__ == "__main__":
    print("Welcome to IntelGuardNet Command Generator!")

    while True:
        print("\nEnter the scan information (or type 'exit' to quit):")
        raw_input = input("Enter scan line: ")  # one line

        # Split it by commas
        parts = [p.strip() for p in raw_input.split(",")]

        # Debug: Print the parts of the input
        print(f"Parts after splitting input: {parts}")

        if len(parts) != 5:
            print("Invalid input! Must be: target, identity, status, open_ports, os")
            continue

        target, identity, status, open_ports, os = parts

        # Create input text for the model
        input_text = f"Target: {target}, Iden: {identity}, Stat: {status}, Open Ports: {open_ports}, OS: {os}"

        # Debug: Print the final input text
        print(f"Input text for model: {input_text}")

        # Get the predicted command
        prediction = predict_command(input_text)

        # Output the prediction
        print(f"\n[Predicted Command]: {prediction}")
