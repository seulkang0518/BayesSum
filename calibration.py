# import os
# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams['axes.grid'] = True
# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.labelsize'] = 26
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
# # plt.rc('font', family='Arial', size=12)
# plt.rc('axes', titlesize=26, labelsize=26, grid=True)
# plt.rc('lines', linewidth=2)
# plt.rc('legend', fontsize=26, frameon=False)
# plt.rc('xtick', labelsize=26, direction='in')
# plt.rc('ytick', labelsize=26, direction='in')
# plt.rc('figure', figsize=(6, 4), dpi=100)

# def load_calibration_npz(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Not found: {path}")
#     data = np.load(path, allow_pickle=True)
#     C_nom   = data.get("C_nom")
#     emp_cov = data.get("emp_cov")
#     t_true  = data.get("t_true")
#     if C_nom is None or emp_cov is None:
#         raise KeyError(f"File {path} missing C_nom/emp_cov")
#     return {
#         "C_nom":   np.asarray(C_nom, dtype=float).ravel(),
#         "emp_cov": np.asarray(emp_cov, dtype=float).ravel(),
#         "t_true":  (float(np.asarray(t_true)) if t_true is not None else None),
#     }

# def plot_calibration_comparison(
#     series,
#     save_dir,
#     main_filename_png="calibration_comparison.png",
#     main_filename_pdf="calibration_comparison.pdf",
#     legend_filename_pdf="calibration_legend.pdf",
#     legend_ncol=3,
#     markers=("o", "s", "^", "D", "v", "p", "X"),
#     linewidth=1.8,
#     markersize=5,
#     figsize=(6.0, 4.5),
#     xlabel="Credible Interval",
#     ylabel="Coverage",
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     main_png = os.path.join(save_dir, main_filename_png)
#     main_pdf = os.path.join(save_dir, main_filename_pdf)
#     legend_pdf = os.path.join(save_dir, legend_filename_pdf)

#     # main figure (no legend)
#     plt.figure(figsize=figsize)
#     handles, labels = [], []
#     for i, (label, payload) in enumerate(series.items()):
#         C_nom = payload["C_nom"]; emp = payload["emp_cov"]
#         if C_nom.shape != emp.shape:
#             raise ValueError(f"Shape mismatch for {label}: {C_nom.shape} vs {emp.shape}")
#         h, = plt.plot(C_nom, emp, f"{markers[i % len(markers)]}-",
#                       linewidth=linewidth, markersize=markersize, label=label)
#         handles.append(h); labels.append(label)

#     # ideal diagonal
#     h_ideal, = plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal")
#     handles.append(h_ideal); labels.append("Ideal")

#     plt.xlabel(xlabel); plt.ylabel(ylabel)
#     plt.title("Calibration")
#     plt.grid(True, ls="--", alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(main_png, dpi=300)
#     plt.savefig(main_pdf, dpi=300)
#     plt.close()

#     # legend-only figure
#     fig_legend, ax_legend = plt.subplots(figsize=(6, 1.2))
#     ax_legend.axis("off")
#     ax_legend.legend(handles, labels, ncol=min(legend_ncol, len(labels)),
#                      loc="center", fontsize=26, frameon=False)
#     plt.savefig(legend_pdf, bbox_inches="tight")
#     plt.close(fig_legend)

#     return {"main_png": main_png, "main_pdf": main_pdf, "legend_pdf": legend_pdf}

# if __name__ == "__main__":
#     # base dir = folder containing this script
#     base_dir = os.path.dirname(os.path.abspath(__file__))

#     # your data lives in ./results relative to this script:
#     data_dir = os.path.join(base_dir, "results")
#     save_dir = data_dir  # save outputs in same results folder

#     # filenames inside ./results — adjust if different
#     files = {
#         "Ising":   os.path.join(data_dir, "ising_calibration_results.npz"),
#         "Potts":   os.path.join(data_dir, "potts_calibration_results.npz"),
#     }

#     # load
#     series = {}
#     for label, path in files.items():
#         try:
#             series[label] = load_calibration_npz(path)
#         except (FileNotFoundError, KeyError) as e:
#             print(f"[Skip] {label}: {e}")

#     if not series:
#         raise SystemExit("No datasets loaded. Check filenames in ./results")

#     # plot
#     outputs = plot_calibration_comparison(
#         series,
#         save_dir=save_dir,
#         main_filename_png="calibration_comparison.png",
#         main_filename_pdf="calibration_comparison.pdf",
#         legend_filename_pdf="calibration_legend.pdf",
#         legend_ncol=4,
#     )
#     print("Saved:", outputs)


import os
import numpy as np
import matplotlib.pyplot as plt

# --- Matplotlib Stylization ---
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 26
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.rc('axes', titlesize=26, labelsize=26, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=22, frameon=True) # Adjusted legend fontsize for better fit
plt.rc('xtick', labelsize=26, direction='in')
plt.rc('ytick', labelsize=26, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

def load_calibration_npz(path):
    """Loads calibration data from an .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    data = np.load(path, allow_pickle=True)
    C_nom   = data["C_nom"]
    emp_cov = data["emp_cov"]
    t_true  = data["t_true"]
    if C_nom is None or emp_cov is None:
        raise KeyError(f"File {path} missing C_nom/emp_cov")
    return {
        "C_nom":   np.asarray(C_nom, dtype=float).ravel(),
        "emp_cov": np.asarray(emp_cov, dtype=float).ravel(),
        "t_true":  (float(np.asarray(t_true)) if t_true is not None else None),
    }

def plot_calibration_comparison(
    series,
    save_dir,
    main_filename_png="calibration_comparison.png",
    main_filename_pdf="calibration_comparison.pdf",
    markers=("o", "s", "^", "D", "v", "p", "X"),
    linewidth=1.8,
    markersize=5,
    figsize=(10, 6), # Slightly adjusted figsize for legend
    xlabel="Credible Interval",
    ylabel="Coverage",
):
    """
    Plots calibration curves and saves the figure with the legend inside.
    """
    os.makedirs(save_dir, exist_ok=True)
    main_png = os.path.join(save_dir, main_filename_png)
    main_pdf = os.path.join(save_dir, main_filename_pdf)

    plt.figure(figsize=figsize)
    for i, (label, payload) in enumerate(series.items()):
        C_nom = payload["C_nom"]
        emp = payload["emp_cov"]
        if C_nom.shape != emp.shape:
            raise ValueError(f"Shape mismatch for {label}: {C_nom.shape} vs {emp.shape}")
        plt.plot(C_nom, emp, f"{markers[i % len(markers)]}-",
                 linewidth=linewidth, markersize=markersize, label=label)

    # Ideal diagonal
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal")

    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.grid(True, ls="--", alpha=0.5)
    plt.title("Calibration", fontsize=32)
    # --- CHANGE: Add legend directly to the plot ---
    # The legend will use the labels provided in the plot() calls.
    # 'loc' places it in the lower right corner, a common spot for calibration plots.
    plt.legend(loc="lower right", fontsize=32)

    plt.tight_layout()
    plt.savefig(main_png, dpi=300)
    plt.savefig(main_pdf, dpi=300)
    plt.close()

    # --- CHANGE: Return only the main figure paths ---
    return {"main_png": main_png, "main_pdf": main_pdf}

if __name__ == "__main__":
    # base dir = folder containing this script
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd() # Fallback for interactive environments

    # Your data lives in ./results relative to this script:
    data_dir = os.path.join(base_dir, "results")
    save_dir = data_dir  # save outputs in the same results folder

    # Create dummy data for demonstration if 'results' folder doesn't exist
    if not os.path.exists(data_dir):
        print("Creating dummy data in './results' for demonstration.")
        os.makedirs(data_dir, exist_ok=True)
        # Dummy data 1
        x1 = np.linspace(0, 1, 10)
        y1 = x1 - 0.05 * np.sin(x1 * np.pi) # Slightly under-confident
        np.savez(os.path.join(data_dir, "ising_calibration_results.npz"), C_nom=x1, emp_cov=y1)
        # Dummy data 2
        x2 = np.linspace(0, 1, 10)
        y2 = x2 + 0.05 * np.sin(x2 * np.pi) # Slightly over-confident
        np.savez(os.path.join(data_dir, "potts_calibration_results.npz"), C_nom=x2, emp_cov=y2)


    # Filenames inside ./results — adjust if different
    files = {
        "Poisson":   os.path.join(data_dir, "poisson_calibration_results.npz"),
        "Potts":   os.path.join(data_dir, "unnormalised_potts_g_calibration_results.npz"),
        # "Potts (MH)":   os.path.join(data_dir, "unnormalised_potts_mh_calibration_results.npz"),
        "Uniform":   os.path.join(data_dir, "potts_calibration_results.npz"),
        "Mixed":   os.path.join(data_dir, "mixed_calibration_results.npz"),
    }

    # Load data
    series = {}
    for label, path in files.items():
        try:
            series[label] = load_calibration_npz(path)
        except (FileNotFoundError, KeyError) as e:
            print(f"[Skip] {label}: {e}")

    if not series:
        raise SystemExit("No datasets loaded. Check filenames in ./results")

    # Plot
    # --- CHANGE: Removed legend-specific arguments from the function call ---
    outputs = plot_calibration_comparison(
        series,
        save_dir=save_dir,
        main_filename_png="calibration_comparison_with_legend.png",
        main_filename_pdf="calibration_comparison_with_legend.pdf",
    )
    print("Saved plots:", outputs)