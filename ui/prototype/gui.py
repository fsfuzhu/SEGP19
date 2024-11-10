import tkinter as tk

def console_output(message):
    console_text.config(state="normal")  # Enable editing to insert text
    console_text.insert("end", message + "\n")  # Insert the message
    console_text.config(state="disabled")  # Disable editing to prevent user input
    console_text.see("end")  # Auto-scroll to the bottom

# Create the main window
root = tk.Tk()
root.title("GUI Layout")
root.state("zoomed")

# Configure the rows and columns of the root window to be resizable
root.grid_rowconfigure(0, weight = 3)
root.grid_rowconfigure(1, weight = 1)
root.grid_columnconfigure(0, weight = 1)
root.grid_columnconfigure(1, weight = 50)

frames = {
    "Input": (0, 0),
    "Console (Debugging)": (1, 0),
    "User View": (0, 1, 2)
}

for label, (row, col, *rowspan) in frames.items():
    # Set frame height for the Console (Debugging) section and create the frame
    height = 100 if label == "Console (Debugging)" else None
    frame = tk.Frame(root, bd=2, relief="solid", height=height)
    frame.grid(row=row, column=col, rowspan=rowspan[0] if rowspan else 1, padx=10, pady=10, sticky="nsew")
    
    # Add console text widget for the debugging section
    if label == "Console (Debugging)":
        console_text = tk.Text(frame, bg="black", fg="white", wrap="word", state="disabled", height=5)
        console_text.pack(expand=True, fill="both")
    else:
        tk.Label(frame, text=label).pack(expand=True)


# Run the application
root.mainloop()
