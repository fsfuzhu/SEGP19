import customtkinter as ctk
import threading

class SplashScreen(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.center_window(530, 430)

        self.label = ctk.CTkLabel(self, text="Starting application...", font=("Arial", 20))
        self.label.place(relx=0.5, rely=0.7, anchor='center')  # Positioned near the bottom middle

        self.progress_bar = ctk.CTkProgressBar(self, orientation='horizontal', mode='determinate')
        self.progress_bar.place(relx=0.5, rely=0.95, anchor='center', relwidth=0.9)  # Full-width progress bar at bottom
        self.progress_bar.set(0)

        self.total_steps = 0

        threading.Thread(target=self.load_main_app, daemon=True).start()

    def center_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def load_main_app(self):
        import time
        import importlib.util

        stages = [
            "Importing core modules...",
            "Loading assets...",
            "Initializing UI...",
            "Finalizing startup..."
        ]

        heavy_imports = [
            "os",
            "cv2",
            "torch",
            "numpy",
            "tkinter",
            "customtkinter",
            "ultralytics",
            "threading",
            "pyvips"
        ]

        self.total_steps = len(stages) + len(heavy_imports)

        # Core Modules
        for i, stage in enumerate(stages[:1]):  
            self.label.configure(text=stage)
            self.progress_bar.set((i + 1) / self.total_steps)
            time.sleep(0.5)

        # Heavy Imports
        for i, module in enumerate(heavy_imports, start=1):
            if importlib.util.find_spec(module):
                try:
                    importlib.import_module(module)
                    self.label.configure(text=f"Loading {module}...")
                except ModuleNotFoundError:
                    self.label.configure(text=f"{module} not found, skipping...")
            else:
                self.label.configure(text=f"{module} not found, skipping...")

            self.progress_bar.set((i + 1) / self.total_steps)

        # Finalizing Step
        self.label.configure(text=stages[-1])  # "Finalizing startup..."
        self.progress_bar.set(1.0)

        from v2 import MainApp
        self.after(0, lambda: self.close_and_launch_main(MainApp))

    def close_and_launch_main(self, MainApp):
        if self.winfo_exists():
            self.destroy()
        main_app = MainApp()
        main_app.mainloop()

    def destroy(self):
        for task in self.tk.call('after', 'info'):
            self.after_cancel(task)
        super().destroy()

if __name__ == "__main__":
    SplashScreen().mainloop()
