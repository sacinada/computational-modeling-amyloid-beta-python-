# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:59:54 2026  

@author: sacin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

"""
We simplified the original equation to some extent, ignoring some variables.
"""
def abeta_dynamics(y, t):
    intra, extra, ao = y

    production = 0.2  # Aβ production rate
    k_deg = 0.05  # Intracellular degradation rate
    k_secrete = 0.1  # Secretion rate to extracellular space
    k_aggregate = 0.01  # Aggregation rate to fibrils
    k_clear = 0.02  # Clearance rate of extracellular Aβ
    k_dissolve = 0.001  # Dissolution rate of amyloid plaques

    # 3 equations
    dintra_dt = production - k_deg * intra - k_secrete * intra
    dextra_dt = k_secrete * intra - k_aggregate * extra - k_clear * extra
    dao_dt = k_aggregate * extra - k_dissolve * ao

    return [dintra_dt, dextra_dt, dao_dt]


days = 500
t = np.linspace(0, days, 1000)

y0 = [0.1, 0.5, 0.0]  # [intracellular Aβ, extracellular Aβ, AO]


solution = odeint(abeta_dynamics, y0, t)
intra_abeta, extra_abeta, ao = solution.T
t_years = t / 365


# Plot
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'Arial',
    'figure.figsize': (12, 8),
    'figure.dpi': 100
})

plt.figure(figsize=(12, 8))

plt.plot(t_years, intra_abeta, 'royalblue', linewidth=2.5,
         label='Intracellular Aβ', linestyle='-')

plt.plot(t_years, extra_abeta, 'darkorange', linewidth=2.5,
         label='Extracellular Aβ', linestyle='-.')

plt.plot(t_years, ao, 'mediumpurple', linewidth=2.5,
         label='AβO', linestyle='--')

# Two critical time
plt.axvline(x=300 / 365, color='red', linestyle='--', alpha=0.7)
plt.text(315 / 365, np.max(ao) * 0.55, 'Amyloid cascade\ntipping point',
         fontsize=11, color='darkred', ha='left')

plt.axvline(x=450 / 365, color='green', linestyle='-.', alpha=0.7)
plt.text(460 / 365, np.max(ao) * 0.35, 'Clinical AD\nthreshold',
         fontsize=11, color='darkgreen')

# Three biological phases
plt.axvspan(0, 100 / 365, alpha=0.1, color='lightblue', zorder=0)
plt.text(50 / 365, np.max(ao) * 0.85, 'Subclinical phase',
         fontsize=11, color='navy', ha='center', va='top')

plt.axvspan(100 / 365, 400 / 365, alpha=0.1, color='gold', zorder=0)
plt.text(250 / 365, np.max(ao) * 0.85, 'Pathological\nprogression',
         fontsize=11, color='darkgoldenrod', ha='center', va='top')

plt.axvspan(400 / 365, 500 / 365, alpha=0.1, color='salmon', zorder=0)
plt.text(480 / 365, np.max(ao) * 0.85, 'Clinical\ndisease',
         fontsize=11, color='darkred', ha='center', va='top')

max_intra = np.max(intra_abeta)
max_intra_time = t_years[np.argmax(intra_abeta)]
max_extra = np.max(extra_abeta)
max_extra_time = t_years[np.argmax(extra_abeta)]
ao_final = ao[-1]


plt.xlabel('Time (years)')
plt.ylabel('Relative Concentration (a.u.)')
plt.title('Dynamics of β-amyloid Species in Alzheimer\'s Pathogenesis', fontsize=16, pad=20)
plt.legend(loc='upper right', fontsize=12, framealpha=0.9)

plt.grid(alpha=0.2, linestyle='--')
plt.xlim(0, np.max(t_years))
plt.ylim(0, np.max(ao) * 1.15)


summary_text = (
    "Key parameters:\n"
    "Production: 0.2 a.u./day\n"
    "Secretion: k=0.1/day\n"
    "Aggregation: k=0.01/day\n"
    r"Clearance: k=0.02/day" "\n"
    "Plaque dissolution: k=0.001/day"
)
plt.gca().text(
    0.02, 0.95, summary_text,
    transform=plt.gca().transAxes,
    fontsize=10, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
)


plt.tight_layout()
plt.savefig('abeta_dynamics.png', dpi=300)
plt.show()


# Memantine Plot
"""
Intracellular Aβ dynamics (Eq. (1)) with/without Memantine
- No drug:   dA_i/dt = [λ_i (1 + R) - d_i * A_i] * (N/N0)
- Memantine: dA_i/dt = [λ_i (1-α) * (1 + R*(1-η)) - d_i*(1+δ) * A_i] * (N/N0)
  α = alpha_mem
  η = eta_mem
  δ = delta_mem
"""
# Without Memantine
lam_i = 0.08      # λ_iβ, baseline production rate (1/time)
R = 0.60          # oxidative stress factor (dimensionless)
d_i = 0.03        # intracellular degradation rate (1/time)
N_over_N0 = 1.0   # neuron density ratio N/N0

# With Memantine
alpha_mem = 0.30  # reduces baseline production fraction (≈20–40%)
eta_mem   = 0.40  # reduces oxidative-stress contribution
delta_mem = 0.40  # increases degradation (≈30–50%)


# Initial condition & time span
A_i0 = 1.0
t_span = (0.0, 200.0)
t_eval = np.linspace(t_span[0], t_span[1], 800)


def rhs_no_drug(t, A_i):
    prod = lam_i * (1.0 + R)
    return (prod - d_i * A_i) * N_over_N0


def rhs_memantine(t, A_i):
    lam_eff = lam_i * (1.0 - alpha_mem)
    R_eff   = R * (1.0 - eta_mem)
    d_eff   = d_i * (1.0 + delta_mem)
    prod = lam_eff * (1.0 + R_eff)
    return (prod - d_eff * A_i) * N_over_N0


sol_no = solve_ivp(rhs_no_drug, t_span, [A_i0], t_eval=t_eval,
                   rtol=1e-8, atol=1e-10)
sol_mem = solve_ivp(rhs_memantine, t_span, [A_i0], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)


plt.figure()
plt.plot(sol_no.t,  sol_no.y[0],  label="No drug")
plt.plot(sol_mem.t, sol_mem.y[0], label="Memantine")
plt.xlabel("Time (a.u.)")
plt.ylabel(r"Intracellular Aβ, $A_\beta^i$ (a.u.)")
plt.title("Intracellular Aβ dynamics: without vs with Memantine")
plt.legend()
plt.tight_layout()
plt.show()


"""
The main references:
https://bmcsystbiol.biomedcentral.com/articles/10.1186/s12918-016-0348-2#Sec1
https://pmc.ncbi.nlm.nih.gov/articles/PMC9138537/#sec3-biomedicines-10-01153
https://pubmed.ncbi.nlm.nih.gov/16906789/
"""