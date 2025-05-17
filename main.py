import customtkinter as ctk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

main = ctk.CTk()
main.geometry("960x540+200+200")
main.title("IntelGuard - Your all-in-one Cyber Security toolkit")

main_frame = ctk.CTkFrame(main)
main_frame.pack(fill="both", expand=True)

# Left frame for vertical buttons with dark grey background
tab_button_bg = "#1f1f1f"
tab_button_frame = ctk.CTkFrame(main_frame, width=160, fg_color=tab_button_bg)
tab_button_frame.pack(side="left", fill="y")

top_button_frame = ctk.CTkFrame(tab_button_frame, fg_color=tab_button_bg)
top_button_frame.pack(side="top", fill="x", anchor="n")

bottom_button_frame = ctk.CTkFrame(tab_button_frame, fg_color=tab_button_bg)
bottom_button_frame.pack(side="bottom", fill="x", anchor="s")

# Content area with slightly lighter dark background
tab_content_bg = "#2a2a2a"
tab_content_frame = ctk.CTkFrame(main_frame, fg_color=tab_content_bg)
tab_content_frame.pack(side="left", fill="both", expand=True)

# Create tabs (frames)
home_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)
vuln_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)
virus_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)
settings_frame = ctk.CTkFrame(tab_content_frame, fg_color=tab_content_bg)

for frame in (home_frame, vuln_frame, virus_frame, settings_frame):
    frame.place(relx=0, rely=0, relwidth=1, relheight=1)

# Tab labels inside frames
ctk.CTkLabel(home_frame, text="Home", font=("Segoe UI", 16), fg_color=tab_content_bg).pack(pady=20)
ctk.CTkLabel(vuln_frame, text="Vulnerabilities Scanning", font=("Segoe UI", 16), fg_color=tab_content_bg).pack(pady=20)
ctk.CTkLabel(virus_frame, text="Virus Scanning", font=("Segoe UI", 16), fg_color=tab_content_bg).pack(pady=20)
ctk.CTkLabel(settings_frame, text="Settings", font=("Segoe UI", 16), fg_color=tab_content_bg).pack(pady=20)

active_button = None

# Colors for buttons
inactive_fg = tab_button_bg        # same as button panel background, so looks transparent
inactive_text = "#cccccc"          # light gray text for inactive buttons
active_fg = "#2a7f2a"              # solid medium green for active button bg
active_text = "#a8f0a8"            # soft green text for active
hover_fg = "#3ba13b"               # lighter green on hover

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
btn_settings = ctk.CTkButton(bottom_button_frame, text="Settings", height=40, corner_radius=0,
                             fg_color=inactive_fg, text_color=inactive_text,
                             hover_color=hover_fg,
                             command=lambda: show_tab(settings_frame, btn_settings))

btn_home.pack(fill="x")
btn_vuln.pack(fill="x")
btn_virus.pack(fill="x")
btn_settings.pack(fill="x")

show_tab(home_frame, btn_home)



LabelHome1 = ctk.CTkLabel(home_frame, text="Welcome to IntelGuard!",
                      font=ctk.CTkFont(size=24, weight="bold"))
LabelHome1.pack(pady = 0)


main.mainloop()
