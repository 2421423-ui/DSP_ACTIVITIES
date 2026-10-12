import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# Core DSP helpers
def sort_by_n(n, y):
    """Sort (n, y) pairs by ascending index n, for clean stem plots."""
    n = np.asarray(n)
    y = np.asarray(y)
    order = np.argsort(n)
    return n[order], y[order]

def compute_all_operations(x, n0, k):
    """
    Given sequence values x (array), its starting index n0, and shift
    amount k, return a dict with the index/value arrays for:
    original, delay, advance, reversal, composite method 1, composite method 2.
    """
    x = np.asarray(x, dtype=float)
    n_x = np.arange(n0, n0 + len(x))

    # 1. Time Delay: y[n] = x[n - k]  ->  n_y = n_x + k, values unchanged
    n_delay, y_delay = sort_by_n(n_x + k, x)

    # 2. Time Advance: y[n] = x[n + k]  ->  n_y = n_x - k, values unchanged
    n_adv, y_adv = sort_by_n(n_x - k, x)

    # 3. Time Reversal: y[n] = x[-n]  ->  n_y = -n_x, values paired as-is
    n_rev, y_rev = sort_by_n(-n_x, x)

    # 4. Composite Method 1 (Shift then Fold):
    #    v[n] = x[n + k]  -> n_v = n_x - k, values = x
    #    y[n] = v[-n]     -> n_y = -n_v = k - n_x, values = x
    n_v1, y_v1 = sort_by_n(n_x - k, x)              # intermediate v[n]
    n_c1, y_c1 = sort_by_n(k - n_x, x)               # final y[n]

    # 5. Composite Method 2 (Fold then Shift):
    #    v[n] = x[-n]      -> n_v = -n_x, values = x
    #    y[n] = v[n - k]   -> n_y = n_v + k = k - n_x, values = x
    n_v2, y_v2 = sort_by_n(-n_x, x)                  # intermediate v[n]
    n_c2, y_c2 = sort_by_n(k - n_x, x)                # final y[n]

    energy = float(np.sum(np.abs(x) ** 2))

    return {
        "original": (n_x, x),
        "delay": (n_delay, y_delay),
        "advance": (n_adv, y_adv),
        "reversal": (n_rev, y_rev),
        "composite1_intermediate": (n_v1, y_v1),
        "composite1_final": (n_c1, y_c1),
        "composite2_intermediate": (n_v2, y_v2),
        "composite2_final": (n_c2, y_c2),
        "energy": energy,
    }


def format_sequence(n, y, label):
    """Return a human-readable multi-line string like the slide examples."""
    lines = [f"{label}:"]
    pairs = ", ".join(f"{label.split('[')[0].strip()}[{int(ni)}]={yi:g}"
                    for ni, yi in zip(n, y))
    lines.append("  " + pairs)
    values_only = "{" + ", ".join(f"{yi:g}" for yi in y) + "}"
    lines.append(f"  Result: {label.split('[')[0].strip()}[n] = {values_only}"
                f"  for n = {int(n[0])}..{int(n[-1])}")
    return "\n".join(lines)


# Tkinter application
class SequenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Discrete-Time Sequence Operations Visualizer")
        self.root.geometry("1280x900")

        # Color Palette
        self.COLOR_BG = "#0f172a"          # Deep Slate / Dark Navy Background
        self.COLOR_CARD = "#1e293b"        # Card Container Background
        self.COLOR_BORDER = "#334155"      # Border lines
        self.COLOR_FG = "#f8fafc"          # White/Light Text
        self.COLOR_MUTED = "#94a3b8"       # Subtle Muted Text
        self.COLOR_PRIMARY = "#2563eb"     # Blue Primary Accent
        self.COLOR_PRIMARY_HOVER = "#1d4ed8"
        self.COLOR_SECONDARY = "#334155"   # Secondary Button
        self.COLOR_SECONDARY_HOVER = "#475569"
        self.COLOR_CYAN = "#38bdf8"        # Cyan Accent for Stem Plots

        self.root.configure(bg=self.COLOR_BG)
        self._apply_custom_theme()

        self._build_input_frame()
        self._build_output_text()
        self._build_plot_area()

        self.compute_and_plot()

    def _apply_custom_theme(self):
        """Configure TTK styles to match dark blue aesthetic."""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure Frame styles
        style.configure("TFrame", background=self.COLOR_BG)
        style.configure("Card.TFrame", background=self.COLOR_CARD, relief="flat")

        # Configure LabelFrames
        style.configure("TLabelframe", background=self.COLOR_CARD, bordercolor=self.COLOR_BORDER, borderwidth=1)
        style.configure("TLabelframe.Label", background=self.COLOR_CARD, foreground=self.COLOR_CYAN, font=("Segoe UI", 10, "bold"))

        # Configure Labels
        style.configure("TLabel", background=self.COLOR_CARD, foreground=self.COLOR_FG, font=("Segoe UI", 9))

        # Configure Entries
        style.configure("TEntry", fieldbackground="#0f172a", foreground=self.COLOR_FG, bordercolor=self.COLOR_BORDER, insertcolor=self.COLOR_FG)
        style.map("TEntry", bordercolor=[("focus", self.COLOR_CYAN)])

        # Configure Buttons
        style.configure("Primary.TButton", background=self.COLOR_PRIMARY, foreground="#ffffff", font=("Segoe UI", 9, "bold"), borderwidth=0, padding=6)
        style.map("Primary.TButton", background=[("active", self.COLOR_PRIMARY_HOVER)])

        style.configure("Secondary.TButton", background=self.COLOR_SECONDARY, foreground=self.COLOR_FG, font=("Segoe UI", 9), borderwidth=0, padding=6)
        style.map("Secondary.TButton", background=[("active", self.COLOR_SECONDARY_HOVER)])

    # ---------------- UI construction ----------------
    def _build_input_frame(self):
        frame = ttk.LabelFrame(self.root, text=" Input Parameters ", padding=10)
        frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=8)

        ttk.Label(frame, text="Sequence x[n] (comma-separated):").grid(
            row=0, column=0, sticky="w", padx=5, pady=5)
        self.seq_entry = ttk.Entry(frame, width=35)
        self.seq_entry.insert(0, "1, 2, 3, 4, 5")
        self.seq_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame, text="Starting Index n0:").grid(
            row=0, column=2, sticky="w", padx=5, pady=5)
        self.n0_entry = ttk.Entry(frame, width=8)
        self.n0_entry.insert(0, "-2")
        self.n0_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(frame, text="Shift Amount k:").grid(
            row=0, column=4, sticky="w", padx=5, pady=5)
        self.k_entry = ttk.Entry(frame, width=8)
        self.k_entry.insert(0, "2")
        self.k_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        compute_btn = ttk.Button(frame, text="Compute & Plot", style="Primary.TButton",
                                command=self.compute_and_plot)
        compute_btn.grid(row=0, column=6, padx=10, pady=5)

        reset_btn = ttk.Button(frame, text="Reset Default", style="Secondary.TButton",
                                command=self.reset_defaults)
        reset_btn.grid(row=0, column=7, padx=5, pady=5)

    def _build_output_text(self):
        frame = ttk.LabelFrame(self.root, text=" Numeric Results ", padding=8)
        frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=4)

        self.output_text = ScrolledText(
            frame, height=10, wrap=tk.WORD, font=("Consolas", 10),
            bg="#0f172a", fg="#38bdf8", insertbackground="#ffffff",
            selectbackground="#2563eb", selectforeground="#ffffff",
            bd=1, relief="solid"
        )
        self.output_text.pack(fill=tk.X, padx=2, pady=2)

    def _build_plot_area(self):
        frame = ttk.LabelFrame(self.root, text=" Visualizations ", padding=8)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Style Matplotlib canvas background to match dark blue card background
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor=self.COLOR_CARD)
        self.axes = self.fig.subplots(2, 3)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().configure(bg=self.COLOR_CARD, highlightthickness=0)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.configure(bg=self.COLOR_CARD)
        for button in self.toolbar.winfo_children():
            button.configure(bg=self.COLOR_CARD)
        self.toolbar.update()

    # ---------------- Actions ----------------
    def reset_defaults(self):
        self.seq_entry.delete(0, tk.END)
        self.seq_entry.insert(0, "1, 2, 3, 4, 5")
        self.n0_entry.delete(0, tk.END)
        self.n0_entry.insert(0, "-2")
        self.k_entry.delete(0, tk.END)
        self.k_entry.insert(0, "2")
        self.compute_and_plot()

    def _parse_inputs(self):
        raw_seq = self.seq_entry.get().strip()
        if not raw_seq:
            raise ValueError("Please enter at least one sequence value.")
        try:
            x = [float(v.strip()) for v in raw_seq.split(",") if v.strip() != ""]
        except ValueError:
            raise ValueError("Sequence values must be numbers separated by commas, "
                            "e.g. 1, 2, 3, 4, 5")
        if len(x) == 0:
            raise ValueError("Please enter at least one sequence value.")

        try:
            n0 = int(self.n0_entry.get().strip())
        except ValueError:
            raise ValueError("Starting index n0 must be an integer.")

        try:
            k = int(self.k_entry.get().strip())
        except ValueError:
            raise ValueError("Shift amount k must be an integer.")

        return x, n0, k

    def compute_and_plot(self):
        try:
            x, n0, k = self._parse_inputs()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        results = compute_all_operations(x, n0, k)
        self._update_text(results, x, n0, k)
        self._update_plots(results, k)

    # ---------------- Rendering ----------------
    def _update_text(self, results, x, n0, k):
        n_x, x_arr = results["original"]
        n_delay, y_delay = results["delay"]
        n_adv, y_adv = results["advance"]
        n_rev, y_rev = results["reversal"]
        n_c1, y_c1 = results["composite1_final"]
        n_c2, y_c2 = results["composite2_final"]

        lines = []
        lines.append(f"Original Sequence: x[n] over n = {n0}..{n0 + len(x) - 1}")
        lines.append("  x[n] = " + "{" + ", ".join(f"{v:g}" for v in x_arr) + "}")
        lines.append("")
        lines.append(format_sequence(n_delay, y_delay, f"1) Time Delay: y[n] = x[n - {k}]"))
        lines.append("")
        lines.append(format_sequence(n_adv, y_adv, f"2) Time Advance: y[n] = x[n + {k}]"))
        lines.append("")
        lines.append(format_sequence(n_rev, y_rev, "3) Time Reversal: y[n] = x[-n]"))
        lines.append("")
        lines.append(format_sequence(
            n_c1, y_c1,
            f"4) Composite Method 1 (Shift then Fold): y[n] = x[-n + {k}]"))
        lines.append("")
        lines.append(format_sequence(
            n_c2, y_c2,
            f"5) Composite Method 2 (Fold then Shift): y[n] = x[-n + {k}]"))
        lines.append("")
        same = np.array_equal(n_c1, n_c2) and np.allclose(y_c1, y_c2)
        lines.append(f"Note: Method 1 and Method 2 give the SAME result "
                    f"({'confirmed match' if same else 'MISMATCH - check inputs'}), "
                    f"as expected since both compute y[n] = x[-n + {k}].")
        lines.append("")
        lines.append(f"Energy of original sequence: E = sum |x[n]|^2 = {results['energy']:g}")

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "\n".join(lines))

    def _update_plots(self, results, k):
        specs = [
            ("original", "Original: x[n]"),
            ("delay", f"1) Time Delay: y[n] = x[n - {k}]"),
            ("advance", f"2) Time Advance: y[n] = x[n + {k}]"),
            ("reversal", "3) Time Reversal: y[n] = x[-n]"),
            ("composite1_final", f"4) Composite M1 (Shift->Fold): y[n]=x[-n+{k}]"),
            ("composite2_final", f"5) Composite M2 (Fold->Shift): y[n]=x[-n+{k}]"),
        ]

        for ax, (key, title) in zip(self.axes.flat, specs):
            n, y = results[key]
            ax.clear()
            ax.set_facecolor("#0f172a")  # Inner plot dark background

            # Custom Stem Plot colors (Sky blue stems, bright cyan markers)
            markerline, stemlines, baseline = ax.stem(n, y, basefmt=" ")
            markerline.set_color(self.COLOR_CYAN)
            markerline.set_markerfacecolor(self.COLOR_CYAN)
            markerline.set_markersize(5)
            stemlines.set_color("#60a5fa")
            stemlines.set_linewidth(1.5)

            ax.set_title(title, fontsize=9, color=self.COLOR_FG, fontweight="bold")
            ax.set_xlabel("n", fontsize=8, color=self.COLOR_MUTED)
            ax.set_ylabel("value", fontsize=8, color=self.COLOR_MUTED)

            ax.tick_params(colors=self.COLOR_MUTED, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(self.COLOR_BORDER)

            ax.grid(True, color=self.COLOR_BORDER, alpha=0.5, linestyle="--")
            ax.axhline(0, color="#64748b", lw=0.8)
            ax.axvline(0, color="#64748b", lw=0.8)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = SequenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
