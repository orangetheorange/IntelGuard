import customtkinter as ctk
from PIL import Image
from tkinter import messagebox
import virusscan
import threading
import json
from tkinter import filedialog
import os
import subprocess
import webbrowser


class Target():
    def __init__(self, name, add):
        self.name = name
        self.add = add

    def scan(self):
        messagebox.showinfo("Info", "Feature in development")


targets = {}


def NewTarget():
    def create_target():
        name = subEntry1.get()
        address = subEntry2.get()
        if name not in targets:
            if name and address:
                targets[name] = Target(name, address)
                create_target_button(name)
                subWin.destroy()
        else:
            messagebox.showwarning("Warning", "The name cannot overlap with an existing target")

    subWin = ctk.CTkToplevel(main)
    subWin.geometry("600x300")
    subWin.title("Create New Target")
    subWin.transient(main)
    subWin.grab_set()
    subWin.lift()

    # Center the window
    main.update_idletasks()
    x = main.winfo_x() + (main.winfo_width() - 600) // 2
    y = main.winfo_y() + (main.winfo_height() - 400) // 2
    subWin.geometry(f"+{x}+{y}")

    subLabel1 = ctk.CTkLabel(subWin, text="Enter the new target name:")
    subLabel1.pack(pady=(20, 5))

    subEntry1 = ctk.CTkEntry(subWin, width=400)
    subEntry1.pack(pady=5)

    subLabel2 = ctk.CTkLabel(subWin, text="Enter the target address (IP/Domain/URL):")
    subLabel2.pack(pady=(20, 5))

    subEntry2 = ctk.CTkEntry(subWin, width=400)
    subEntry2.pack(pady=5)

    create_btn = ctk.CTkButton(subWin, text="Create Target", command=create_target, fg_color="#2a2a2a",
                               hover_color="#3a3a3a")
    create_btn.pack(pady=20)


def create_target_button(target_name):
    outline_frame = ctk.CTkFrame(scrollable_frame, fg_color="#4a4a4a", corner_radius=8, border_width=2,
                                 border_color="#5a9d5a")
    outline_frame.pack(fill="x", pady=5, padx=0)

    btn = ctk.CTkButton(
        outline_frame,
        text=target_name,
        height=80,
        corner_radius=5,
        fg_color="#2a2a2a",
        hover_color="#3a3a3a",
        command=lambda: target_button_function(target_name),
        font=ctk.CTkFont(size=40, weight="bold")
    )
    btn.pack(fill="x", expand=True, padx=2, pady=2)

    targets_canvas.configure(scrollregion=targets_canvas.bbox("all"))


def target_button_function(target_name):
    tar = targets[target_name]

    def start():
        tar.scan()

    subWin2 = ctk.CTkToplevel(main)
    subWin2.geometry("200x120+300+300")
    subWin2.title("Run the scan")
    subLabel1 = ctk.CTkLabel(subWin2, text=f"Target name: {tar.name}")
    subLabel2 = ctk.CTkLabel(subWin2, text=f"Target address: {tar.add}\n\n")
    subButton = ctk.CTkButton(subWin2, text="Start", font=ctk.CTkFont(size=24, weight="bold"), command=start,
                              fg_color="#2a2a2a", hover_color="#3a3a3a")
    subLabel1.pack()
    subLabel2.pack()
    subButton.pack()
    subWin2.transient(main)
    subWin2.grab_set()
    subWin2.lift()


# Initialize main window
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

main = ctk.CTk()
main.geometry("960x540+200+200")
main.title("IntelGuard - Your all-in-one Cyber Security toolkit")

main_frame = ctk.CTkFrame(main)
main_frame.pack(fill="both", expand=True)
main.resizable(False, False)

# Left frame for vertical buttons
tab_button_bg = "#1f1f1f"
tab_button_frame = ctk.CTkFrame(main_frame, width=160, fg_color=tab_button_bg)
tab_button_frame.pack(side="left", fill="y")

# Logo frame
logo_frame = ctk.CTkFrame(tab_button_frame, height=80, fg_color=tab_button_bg)
logo_frame.pack(side="top", fill="x", pady=(0, 10))

try:
    original_image = Image.open("logo_small.png")
    original_width, original_height = original_image.size
    new_width = 80
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    logo_img = ctk.CTkImage(light_image=original_image,
                            dark_image=original_image,
                            size=(new_width, new_height))
    logo_label = ctk.CTkLabel(logo_frame,
                              image=logo_img,
                              text="",
                              fg_color=tab_button_bg)
    logo_label.pack(padx=5, pady=5)
except Exception as e:
    print(f"Error loading logo: {e}")
    logo_label = ctk.CTkLabel(logo_frame,
                              text="IntelGuard",
                              font=("Arial", 12, "bold"),
                              fg_color=tab_button_bg)
    logo_label.pack(pady=10)

top_button_frame = ctk.CTkFrame(tab_button_frame, fg_color=tab_button_bg)
top_button_frame.pack(side="top", fill="x", anchor="n")

bottom_button_frame = ctk.CTkFrame(tab_button_frame, fg_color=tab_button_bg)
bottom_button_frame.pack(side="bottom", fill="x", anchor="s")

# Tab content area
tab_content_bg = "#2a2a2a"
tab_content_frame = ctk.CTkFrame(main_frame, fg_color=tab_content_bg)
tab_content_frame.pack(side="left", fill="both", expand=True)

# Create frames for each tab
home_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)
vuln_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)
virus_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)
settings_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)
console_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)

for frame in (home_frame, vuln_frame, virus_frame, settings_frame, console_frame):
    frame.place(relx=0, rely=0, relwidth=1, relheight=1)

# Home Tab Content
LabelHome1 = ctk.CTkLabel(home_frame, text="Welcome to IntelGuard!\n\n\n",
                          font=ctk.CTkFont(size=24, weight="bold"))
LabelHome1.pack(pady=0)
LabelHome2 = ctk.CTkLabel(home_frame, text="""
To scan vulnerabilities for a specified target, go to the \"Vulnerabilities Scanning\" tab. \n\n To scan the suspiciousness level of a exe file, go the the \"Virus Scanning\" tab. \n\n Go to the \"Settings\" tab for more settings. \n\n\n\n\n\n\n\n If you need more help, please contact the developer: orange.yichengyu.psn@gmail.com""",
                          font=ctk.CTkFont(size=18))
LabelHome2.pack(pady=1)

# Vulnerabilities Tab Content
vuln_content_frame = ctk.CTkFrame(vuln_frame, fg_color=tab_content_bg)
vuln_content_frame.pack(fill="both", expand=True, padx=10, pady=10)

ButtonVuln1 = ctk.CTkButton(
    vuln_content_frame,
    text="+ New Target",
    command=NewTarget,
    fg_color="#2a2a2a",
    hover_color="#3a3a3a",
    height=50
)
ButtonVuln1.pack(pady=(0, 5), fill="x")

separator = ctk.CTkFrame(vuln_content_frame, height=2, fg_color="#3a3a3a")
separator.pack(fill="x", pady=(0, 10))

# Scrollable frame setup for targets
targets_canvas = ctk.CTkCanvas(vuln_content_frame, bg=tab_content_bg, highlightthickness=0)
scrollbar = ctk.CTkScrollbar(vuln_content_frame, orientation="vertical", command=targets_canvas.yview)
scrollable_frame = ctk.CTkFrame(targets_canvas, fg_color=tab_content_bg)

scrollable_frame.bind(
    "<Configure>",
    lambda e: targets_canvas.configure(
        scrollregion=targets_canvas.bbox("all")
    )
)

targets_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", tags="frame")
targets_canvas.configure(yscrollcommand=scrollbar.set)

targets_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


def on_canvas_resize(event):
    targets_canvas.itemconfig("frame", width=event.width)


targets_canvas.bind("<Configure>", on_canvas_resize)

# Virus Scanning Tab Content
virus_content_frame = ctk.CTkFrame(virus_frame, fg_color=tab_content_bg)
virus_content_frame.pack(fill="both", expand=True, padx=20, pady=20)

ctk.CTkLabel(virus_content_frame,
             text="Virus Scanning",
             font=("Segoe UI", 20, "bold"),
             fg_color=tab_content_bg).pack(pady=(0, 20))

instruction_label = ctk.CTkLabel(
    virus_content_frame,
    text="Click the scan button to check an executable file for malicious content",
    font=("Segoe UI", 14),
    fg_color=tab_content_bg
)
instruction_label.pack(pady=(0, 20))

scan_button = ctk.CTkButton(
    virus_content_frame,
    text="Scan File",
    command=lambda: start_virus_scan(),
    fg_color="#2a2a2a",
    hover_color="#3a3a3a",
    height=40
)
scan_button.pack(pady=(0, 20))

result_frame = ctk.CTkFrame(virus_content_frame, fg_color=tab_content_bg)


def start_virus_scan():
    scan_button.configure(state="disabled")
    virus_frame.update()

    def scan_thread():
        try:
            result = virusscan.run_prediction_pipeline(main)
            show_results(result)
        except Exception as e:
            messagebox.showerror("Error", f"Scan failed: {str(e)}")
        finally:
            scan_button.configure(state="normal")

    threading.Thread(target=scan_thread, daemon=True).start()


def show_results(percentage):
    for widget in result_frame.winfo_children():
        widget.destroy()

    ctk.CTkLabel(
        result_frame,
        text="Scan Results",
        font=("Segoe UI", 16, "bold"),
        fg_color=tab_content_bg
    ).pack(pady=(0, 10))

    threat_level = "High" if percentage >= 60 else "Medium" if percentage >= 30 else "Low"
    color = "#ff4d4d" if percentage >= 60 else "#ffcc00" if percentage >= 30 else "#5a9d5a"

    gauge_frame = ctk.CTkFrame(result_frame, fg_color=tab_content_bg)
    gauge_frame.pack(pady=(10, 20))

    gauge_bg = ctk.CTkFrame(gauge_frame, width=400, height=40, fg_color="#1f1f1f", corner_radius=20)
    gauge_bg.pack()

    fill_width = int(400 * (percentage / 100))
    gauge_fill = ctk.CTkFrame(gauge_bg, width=fill_width, height=36, fg_color=color, corner_radius=18)
    gauge_fill.place(relx=0, rely=0, x=2, y=2)

    gauge_text = ctk.CTkLabel(
        gauge_bg,
        text=f"{percentage:.1f}%",
        font=("Segoe UI", 14, "bold"),
        fg_color="transparent"
    )
    gauge_text.place(relx=0.5, rely=0.5, anchor="center")

    ctk.CTkLabel(
        result_frame,
        text=f"Threat Level: {threat_level}",
        font=("Segoe UI", 14),
        text_color=color,
        fg_color=tab_content_bg
    ).pack(pady=(0, 20))

    recommendation = (
        "This file is highly likely to be malicious. Do not execute it!" if percentage >= 60 else
        "This file shows some suspicious characteristics. Use with caution." if percentage >= 30 else
        "This file appears to be safe, but always be cautious with unknown executables."
    )

    ctk.CTkLabel(
        result_frame,
        text=recommendation,
        font=("Segoe UI", 12),
        wraplength=400,
        justify="left",
        fg_color=tab_content_bg
    ).pack(pady=(0, 20))

    ctk.CTkButton(
        result_frame,
        text="Scan Another File",
        command=lambda: [result_frame.pack_forget(), start_virus_scan()],
        fg_color="#2a2a2a",
        hover_color="#3a3a3a"
    ).pack(pady=(20, 0))

    result_frame.pack(pady=(20, 0), fill="x")


# Console Tab Content
current_shell = "cmd"  # Default shell

console_content_frame = ctk.CTkFrame(console_frame, fg_color=tab_content_bg)
console_content_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Console output area
console_output = ctk.CTkTextbox(console_content_frame,
                                wrap="word",
                                font=("Consolas", 12),
                                fg_color="#1e1e1e",
                                text_color="#ffffff")
console_output.pack(fill="both", expand=True, pady=(0, 10))
console_output.insert("end", "IntelGuard Console - Type commands below\n" + "=" * 40 + "\n\n")
console_output.configure(state="disabled")

# Command input area
command_frame = ctk.CTkFrame(console_content_frame, fg_color=tab_content_bg)
command_frame.pack(fill="x", pady=(0, 10))

command_prompt = ctk.CTkLabel(command_frame,
                              text=">>> ",
                              font=("Consolas", 12),
                              width=40)
command_prompt.pack(side="left")

command_entry = ctk.CTkEntry(command_frame,
                             font=("Consolas", 12),
                             fg_color="#1e1e1e",
                             border_width=0)
command_entry.pack(side="left", fill="x", expand=True)
command_entry.bind("<Return>", lambda e: execute_console_command())


def execute_console_command():
    command = command_entry.get()
    if not command.strip():
        return

    # Add command to output
    console_output.configure(state="normal")
    console_output.insert("end", f">>> {command}\n")

    try:
        # Prepare the command based on selected shell
        if current_shell == "powershell":
            full_command = f"powershell -Command \"{command}\""
        else:
            full_command = f"cmd /c \"{command}\""

        # Execute command and capture output
        process = subprocess.Popen(full_command,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        stdout, stderr = process.communicate()

        if stdout:
            console_output.insert("end", stdout)
        if stderr:
            console_output.insert("end", stderr, "error")

    except Exception as e:
        console_output.insert("end", f"Error: {str(e)}\n", "error")

    console_output.see("end")
    console_output.configure(state="disabled")
    command_entry.delete(0, "end")


# Settings Tab Content
settings_content_frame = ctk.CTkFrame(settings_frame, fg_color=tab_content_bg)
settings_content_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Console Settings Section
console_settings_frame = ctk.CTkFrame(settings_content_frame, fg_color=tab_content_bg)
console_settings_frame.pack(fill="x", pady=(0, 20))


def open_nmap_website(event):
    webbrowser.open_new("https://nmap.org/download.html")

nmap_link = ctk.CTkLabel(
    home_frame,
    text="Download Nmap here",
    font=ctk.CTkFont(size=16, underline=True),
    text_color="#3399ff",
    cursor="hand2",
    fg_color=tab_content_bg
)
nmap_link.pack(pady=(10, 20))
nmap_link.bind("<Button-1>", open_nmap_website)


ctk.CTkLabel(console_settings_frame,
             text="Console Settings",
             font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

# Shell selection
shell_frame = ctk.CTkFrame(console_settings_frame, fg_color="transparent")
shell_frame.pack(fill="x", pady=(5, 0))

ctk.CTkLabel(shell_frame,
             text="Default shell:").pack(side="left")
shell_option = ctk.CTkOptionMenu(shell_frame,
                                 values=["Command Prompt (cmd)", "PowerShell"],
                                 command=lambda x: update_shell_preference())
shell_option.pack(side="left", padx=5)
shell_option.set("Command Prompt (cmd)")


def update_shell_preference():
    """Update the shell preference based on user selection"""
    global current_shell
    selection = shell_option.get()
    if "PowerShell" in selection:
        current_shell = "powershell"
    else:
        current_shell = "cmd"


# Report Settings Section
report_frame = ctk.CTkFrame(settings_content_frame, fg_color=tab_content_bg)
report_frame.pack(fill="x", pady=(0, 20))

ctk.CTkLabel(report_frame,
             text="Report Settings",
             font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

# Report format selection
format_frame = ctk.CTkFrame(report_frame, fg_color="transparent")
format_frame.pack(fill="x", pady=(5, 0))

ctk.CTkLabel(format_frame,
             text="Default format:").pack(side="left")
report_format = ctk.CTkOptionMenu(format_frame,
                                  values=["Text", "HTML", "PDF"])
report_format.pack(side="left", padx=5)
report_format.set("HTML")


# Report save location
def choose_report_location():
    folder = filedialog.askdirectory()
    if folder:
        report_location_entry.delete(0, "end")
        report_location_entry.insert(0, folder)


report_location_frame = ctk.CTkFrame(report_frame, fg_color="transparent")
report_location_frame.pack(fill="x", pady=(10, 0))

ctk.CTkLabel(report_location_frame,
             text="Save location:").pack(side="left")
report_location_entry = ctk.CTkEntry(report_location_frame)
report_location_entry.pack(side="left", fill="x", expand=True, padx=5)
report_location_entry.insert(0, os.path.expanduser("~/Documents/IntelGuard_Reports"))
ctk.CTkButton(report_location_frame,
              text="Browse...",
              width=80,
              command=choose_report_location).pack(side="left")


# Save Settings Button
def save_settings():
    settings = {
        'report_format': report_format.get(),
        'report_location': report_location_entry.get(),
        'preferred_shell': shell_option.get()
    }

    try:
        with open('intelguard_config.json', 'w') as f:
            json.dump(settings, f)
        messagebox.showinfo("Success", "Settings saved successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save settings: {str(e)}")


save_btn = ctk.CTkButton(settings_content_frame,
                         text="Save All Settings",
                         fg_color="#2a2a2a",
                         hover_color="#3a3a3a",
                         command=save_settings)
save_btn.pack(pady=(20, 0))

# Tab Navigation
active_button = None
inactive_fg = tab_button_bg
inactive_text = "#cccccc"
active_fg = "#2a7f2a"
active_text = "#a8f0a8"
hover_fg = "#3ba13b"


def reset_button_style(button):
    button.configure(fg_color=inactive_fg, text_color=inactive_text, hover_color=hover_fg)


def set_active_style(button):
    button.configure(fg_color=active_fg, text_color=active_text, hover_color=active_fg)


def show_tab(tab, button):
    global active_button
    tab.lift()
    if active_button:
        reset_button_style(active_button)
    set_active_style(button)
    active_button = button


# Create navigation buttons
btn_home = ctk.CTkButton(top_button_frame, text="Home", height=40, corner_radius=0,
                         fg_color=inactive_fg, text_color=inactive_text,
                         hover_color=hover_fg,
                         command=lambda: show_tab(home_frame, btn_home))
btn_vuln = ctk.CTkButton(top_button_frame, text="Vulnerabilities", height=40, corner_radius=0,
                         fg_color=inactive_fg, text_color=inactive_text,
                         hover_color=hover_fg,
                         command=lambda: show_tab(vuln_frame, btn_vuln))
btn_virus = ctk.CTkButton(top_button_frame, text="Virus Scan", height=40, corner_radius=0,
                          fg_color=inactive_fg, text_color=inactive_text,
                          hover_color=hover_fg,
                          command=lambda: show_tab(virus_frame, btn_virus))
btn_console = ctk.CTkButton(top_button_frame, text="Console", height=40, corner_radius=0,
                            fg_color=inactive_fg, text_color=inactive_text,
                            hover_color=hover_fg,
                            command=lambda: show_tab(console_frame, btn_console))
btn_settings = ctk.CTkButton(bottom_button_frame, text="Settings", height=40, corner_radius=0,
                             fg_color=inactive_fg, text_color=inactive_text,
                             hover_color=hover_fg,
                             command=lambda: show_tab(settings_frame, btn_settings))

btn_home.pack(fill="x")
btn_vuln.pack(fill="x")
btn_virus.pack(fill="x")
btn_console.pack(fill="x")
btn_settings.pack(fill="x")


# Load saved settings
def load_settings():
    global current_shell
    try:
        if os.path.exists('intelguard_config.json'):
            with open('intelguard_config.json', 'r') as f:
                settings = json.load(f)

                # Report settings
                if 'report_format' in settings:
                    report_format.set(settings['report_format'])
                if 'report_location' in settings:
                    report_location_entry.delete(0, "end")
                    report_location_entry.insert(0, settings['report_location'])

                # Console settings
                if 'preferred_shell' in settings:
                    shell_option.set(settings['preferred_shell'])
                    if "PowerShell" in settings['preferred_shell']:
                        current_shell = "powershell"
                    else:
                        current_shell = "cmd"
    except Exception as e:
        print(f"Error loading settings: {e}")


# Load settings when starting
load_settings()

# Start with Home tab active
show_tab(home_frame, btn_home)

main.mainloop()