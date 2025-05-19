import pefile
import math
import numpy as np
import onnxruntime as rt
import joblib
from tkinter import filedialog, ttk
import tkinter as tk
import time

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

    file_size = pe.__data__.size()
    num_sections = len(pe.sections)
    entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint

    imports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                imports.append(imp.name)
    num_imports = len(imports)
    entropy = calculate_entropy(pe.__data__)

    return [file_size, num_sections, entry_point, num_imports, entropy]

def predict_suspiciousness(features):
    X = np.array(features).reshape(1, -1)
    selector = joblib.load("selector.pkl")
    scaler = joblib.load("scaler.pkl")

    X_selected = selector.transform(X)
    X_scaled = scaler.transform(X_selected)

    sess = rt.InferenceSession("linear_regression.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

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

def run_prediction_pipeline():
    path = ask_exe_file()
    if not path:
        print("No file selected.")
        return

    # Create scanning progress window
    progress_root = tk.Toplevel()
    progress_root.title("Scanning...")
    progress_root.geometry("300x80")
    tk.Label(progress_root, text="Analyzing file...").pack(pady=10)

    progress = ttk.Progressbar(progress_root, orient="horizontal", mode="indeterminate", length=250)
    progress.pack(pady=5)
    progress.start()

    # Update GUI so it displays immediately
    progress_root.update()

    # Delay for better visual feedback
    time.sleep(0.5)

    # Run prediction
    try:
        features = get_exe_info(path)
        suspiciousness = predict_suspiciousness(features)
    except Exception as e:
        print("Error during prediction:", e)
        progress_root.destroy()
        return

    progress.stop()
    progress_root.destroy()

    return round(suspiciousness, 0)
