import pefile
import math
import numpy as np
import onnxruntime as rt
import joblib
from tkinter import filedialog
import tkinter as tk
import time
import customtkinter as ctk

def calculate_entropy(data):
    if not data:
        return 0.0
    entropy = 0
    length = len(data)
    freq = [0] * 256
    for byte in data:
        freq[byte if isinstance(byte, int) else byte[0]] += 1
    for f in freq:
        if f:
            p = f / length
            entropy -= p * math.log2(p)
    return entropy

def get_exe_info(file_path):
    pe = pefile.PE(file_path)

    # File size in bytes
    file_size = pe.__data__.size()

    # Number of Sections
    num_sections = len(pe.sections)

    # Entry Point (RVA)
    entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint

    # Number of Imports
    imports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                imports.append(imp.name)
    num_imports = len(imports)

    # Entropy of the whole file
    entropy = calculate_entropy(pe.__data__)
    return [file_size, num_sections, entry_point, num_imports, entropy]

def predict_suspiciousness(features):
    # Convert features to 2D array
    X = np.array(features).reshape(1, -1)

    # Load selector and scaler
    selector = joblib.load("selector.pkl")
    scaler = joblib.load("scaler.pkl")

    # Feature selection
    X_selected = selector.transform(X)

    # Feature scaling
    X_scaled = scaler.transform(X_selected)

    # Load ONNX model
    sess = rt.InferenceSession("linear_regression.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Run prediction
    pred = sess.run([output_name], {input_name: X_scaled.astype(np.float32)})
    suspiciousness = pred[0][0].item()

    return suspiciousness

def ask_exe_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        defaultextension=".exe",
        filetypes=[("Executable files", "*.exe"), ("All files", "*.*")],
        title="Open your exe file"
    )
    return path

def run_prediction_pipeline(parent_window=None):
    path = ask_exe_file()
    if not path:
        print("No file selected.")
        return

    # Create progress popup
    progress_root = ctk.CTkToplevel(parent_window)
    progress_root.geometry("320x120")
    progress_root.title("Scanning...")
    progress_root.attributes("-topmost", True)

    label = ctk.CTkLabel(progress_root, text="Scanning executable...", font=("Arial", 14))
    label.pack(pady=10)

    progress = ctk.CTkProgressBar(progress_root, mode="indeterminate", width=250)
    progress.pack(pady=10)
    progress.start()

    progress_root.update_idletasks()

    # Scan and predict
    try:
        # Add short sleep to show animation a bit before scanning
        time.sleep(0.5)

        features = get_exe_info(path)
        suspiciousness = predict_suspiciousness(features)
        time.sleep(0.3)  # slight delay for UI smoothness
    except Exception as e:
        print("Error:", e)
        progress_root.destroy()
        return

    progress.stop()
    progress_root.destroy()

    return round(suspiciousness, 0)

