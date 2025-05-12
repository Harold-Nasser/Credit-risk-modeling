import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import root_scalar
################################################################################
#1]
# Maturités et rendements observés
maturities = np.array([0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 30])
yields = np.array([4.50, 4.53, 4.59, 4.66, 4.73, 4.81, 5.01, 5.28, 5.42, 5.55, 5.67, 5.77])

# Chargement des données NSS
raw_nss = pd.read_csv("C:\\Fixed income TP2\\feds200628.csv", skiprows=9)
nss = raw_nss[raw_nss["Date"] == "2025-02-28"].iloc[:, [1, 2, 3, 4, 98, 99]]


# Fonction NSS
def NSS(params, maturities):
    a, b, c, d, tau, theta = params
    t = np.array(maturities)
    term1 = a
    term2 = b * (1 - np.exp(-t / tau)) / (t / tau)
    term3 = c * (((1 - np.exp(-t / tau)) / (t / tau)) - np.exp(-t / tau))
    term4 = d * (((1 - np.exp(-t / theta)) / (t / theta)) - np.exp(-t / theta))
    return term1 + term2 + term3 + term4

def ZCY_t(df, maturities):
    zcy_nss = np.full((len(maturities), len(df)), np.nan)
    for i in range(len(df)):
        params = df.iloc[i].to_numpy()
        zcy_nss[:, i] = NSS(params, maturities)
    return zcy_nss

zcy_t = ZCY_t(nss, maturities).flatten()


df_plot = pd.DataFrame({
    "Maturity": maturities,
    "Rendements AA": yields,
    "Taux sans risque": zcy_t,
    "Credit_Spread": (yields - zcy_t)*1000
})

# Premier graphique : yields vs rf
df_yields_rf = df_plot[["Maturity", "Rendements AA", "Taux sans risque"]].melt(id_vars="Maturity", 
                                                          value_vars=["Rendements AA", "Taux sans risque"],
                                                          var_name="Type", 
                                                          value_name="Rate")

plt.figure(figsize=(10, 5))
sns.lineplot(data=df_yields_rf, x="Maturity", y="Rate", hue="Type", linewidth=2)
plt.title("Rendements AA vs Taux sans risque")
plt.xlabel("Échéance (années)")
plt.ylabel("Taux (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Deuxième graphique : écart de crédit
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_plot, x="Maturity", y="Credit_Spread", color="red", linewidth=2)
plt.title("Écart de crédit")
plt.xlabel("Échéance (années)")
plt.ylabel("Écart (bps)")
plt.grid(True)
plt.tight_layout()
plt.show()




################################################################################
#2]

################################################################################
#3]

# Fonction CIR pour les taux zéro-coupon
def zcy_CIR(t, kappa, theta, sigma, r0):
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    B = (2 * np.exp(gamma * t)) / ((gamma + kappa) * (np.exp(gamma * t) - 1) + 2 * gamma)
    A = ((2 * gamma * np.exp((gamma + kappa) * t / 2)) /
         ((gamma + kappa) * (np.exp(gamma * t) - 1) + 2 * gamma))**(2 * kappa * theta / sigma**2)
    return - (np.log(A) - B * r0) / t

# Fonction objectif pour la calibration
def CIR_objective_zcy(params, T_obs, zcy_obs):
    kappa, theta, sigma, r0 = params
    if kappa <= 0 or theta <= 0 or sigma <= 0 or 2 * kappa * theta < sigma**2:
        return 1e6
    zcy_model = np.array([zcy_CIR(t, kappa, theta, sigma, r0) for t in T_obs])
    return np.sum((zcy_model - zcy_obs)**2)

# Données observées
rf_obs = zcy_t / 100
params_init_CIR = [0.5, 0.045, 0.01, 0.011]

# Calibration
res_CIR = minimize(
    CIR_objective_zcy,
    params_init_CIR,
    args=(maturities, rf_obs),
    method="L-BFGS-B",
    bounds=[(1e-4, 10), (1e-4, 1), (1e-4, 1), (1e-4, 1)]
)

params_calib_riskfree = res_CIR.x
kappa_r, theta_r, sigma_r, r0 = params_calib_riskfree

# Taux modélisés
zcy_fitted = np.array([zcy_CIR(t, kappa_r, theta_r, sigma_r, r0) for t in maturities])

# Tracé des résultats
plt.figure(figsize=(10, 6))
plt.plot(maturities, rf_obs, 'o', label="ZCY observés", color="blue")
plt.plot(maturities, zcy_fitted, '-', label="ZCY CIR", color="red")
plt.title("Calibrage Taux sans risque")
plt.xlabel("Maturité (années)")
plt.ylabel("Taux zéro-coupon (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Fonction CIR standard
def CIR(t, kappa, theta, sigma):
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    B = (2 * np.exp(gamma * t)) / ((gamma + kappa) * (np.exp(gamma * t) - 1) + 2 * gamma)
    A = ((2 * gamma * np.exp((gamma + kappa) * t / 2)) /
         ((gamma + kappa) * (np.exp(gamma * t) - 1) + 2 * gamma))**(2 * kappa * theta / sigma**2)
    return {"a": A, "b": B}

# Fonction CIR ajustée (pour r_t avec beta)
def CIR_adjusted(t, kappa, theta, sigma, beta):
    adjusted_theta = (1 + beta) * theta
    adjusted_sigma = sigma * np.sqrt(1 + beta)
    gamma = np.sqrt(kappa**2 + 2 * adjusted_sigma**2)
    B = (2 * np.exp(gamma * t)) / ((gamma + kappa) * (np.exp(gamma * t) - 1) + 2 * gamma)
    A = ((2 * gamma * np.exp((gamma + kappa) * t / 2)) /
         ((gamma + kappa) * (np.exp(gamma * t) - 1) + 2 * gamma))**(2 * kappa * adjusted_theta / adjusted_sigma**2)
    return {"a": A, "b": B}

# ZCY sous le modèle Duffee
def zcy_duffee(t, params):
    kappa_r = params["kappa_r"]
    theta_r = params["theta_r"]
    sigma_r = params["sigma_r"]
    r0      = params["r0"]

    kappa_s = params["kappa_s"]
    theta_s = params["theta_s"]
    sigma_s = params["sigma_s"]
    s0      = params["s0"]

    alpha   = params["alpha"]
    beta    = params["beta"]

    cir1 = CIR_adjusted(t, kappa_r, theta_r, sigma_r, beta)
    cir2 = CIR(t, kappa_s, theta_s, sigma_s)

    return (alpha * t - np.log(cir1["a"]) + cir1["b"] * (1 + beta) * r0
            - np.log(cir2["a"]) + cir2["b"] * s0) / t

# Fonction objectif pour l’optimisation
def duffee_objective(params_vec, T_obs, zcy_obs):
    keys = ["kappa_s", "theta_s", "sigma_s", "s0", "alpha", "beta"]
    params_dict = dict(zip(keys, params_vec))
    params_dict.update({
        "kappa_r": kappa_r,
        "theta_r": theta_r,
        "sigma_r": sigma_r,
        "r0": r0
    })

    zcy_model = np.array([zcy_duffee(t, params_dict) for t in T_obs])
    return np.sum((zcy_model - zcy_obs)**2)

# Données observées (ZCY AA)
zcy_AA = yields / 100  # yields doit être un array NumPy
T_obs = maturities     # maturities aussi

# Grille d'initialisation
grid_inits = [
    [0.3, 0.02, 0.05, 0.005, 0.01, -0.1],
    [0.2, 0.015, 0.04, 0.002, 0.02, -0.5],
    [0.6, 0.025, 0.06, 0.007, 0.005, -0.3]
]

# Bornes pour chaque paramètre
bounds = [
    (0.01, 2),     # kappa_s
    (1e-4, 0.06),  # theta_s
    (1e-4, 0.2),   # sigma_s
    (1e-4, 0.05),  # s0
    (-0.05, 0.05), # alpha
    (-1, 0)        # beta
]

# Recherche du meilleur fit
best_result = None
lowest_loss = np.inf

for init in grid_inits:
    res = minimize(
        duffee_objective,
        x0=init,
        args=(T_obs, zcy_AA),
        bounds=bounds,
        method="L-BFGS-B"
    )
    if res.fun < lowest_loss:
        best_result = res
        lowest_loss = res.fun

params_duffee = dict(zip(
    ["kappa_s", "theta_s", "sigma_s", "s0", "alpha", "beta"],
    best_result.x
))
params_duffee.update({
    "kappa_r": kappa_r,
    "theta_r": theta_r,
    "sigma_r": sigma_r,
    "r0": r0
})

# Courbe modélisée
zcy_model = np.array([zcy_duffee(t, params_duffee) for t in T_obs])

# Tracé du résultat
plt.figure(figsize=(10, 6))
plt.plot(T_obs, zcy_AA, 'o', label="ZCY observés (AA)", color="blue")
plt.plot(T_obs, zcy_model, '-', label="ZCY modélisés (Duffee)", color="red")
plt.xlabel("Maturité (années)")
plt.ylabel("Taux zéro-coupon")
plt.title("Calibrage des rendements AA")
plt.legend()
plt.grid(True)
plt.show()

################################################################################
#4]

# Paramètres calibrés 

# kappa_r 
# theta_r
# sigma_r
# r0

kappa_s = params_duffee["kappa_s"]
theta_s = params_duffee["theta_s"]
sigma_s = params_duffee["sigma_s"]
s0 = params_duffee["s0"]

alpha = params_duffee["alpha"]
beta = params_duffee["beta"]

# Discrétisation
delta_t = 1/400
echeance = 5
N_t = int(echeance / delta_t) + 1

delta_r = 0.001
r_min = 0
r_max = 0.1
N_r = int((r_max - r_min) / delta_r) + 1

delta_s = 0.0005
s_min = 0
s_max = 0.018
N_s = int((s_max - s_min) / delta_s) + 1

# Valeur nominale et coupon
M = 1_000_000
C = 0.04 * M / 2  # Semestriel

# Grille 3D g(t, r, s)
g = np.zeros((N_t, N_r, N_s))
g[-1, :, :] = M + C  # À l’échéance

# Boucle en temps inversé
for t in reversed(range(N_t - 1)):
    for i in range(1, N_r - 1):
        for j in range(1, N_s - 1):

            A_ij = (1 - (delta_t * (sigma_r**2) * i / delta_r)
                      - (delta_t * (sigma_s**2) * j / delta_s)
                      - delta_t * ((1 + beta) * delta_r * i + alpha + delta_s * j))

            B_ijp1 = delta_t * (((kappa_s * theta_s + sigma_s**2 * j) / (2 * delta_s)) - (kappa_s * j / 2))
            B_ijm1 = delta_t * (((sigma_s**2 * j - kappa_s * theta_s) / (2 * delta_s)) + (kappa_s * j / 2))

            C_ip1j = delta_t * (((kappa_r * theta_r + (sigma_r**2) * i) / (2 * delta_r)) - (kappa_r * i / 2))
            C_im1j = delta_t * ((((sigma_r**2) * i - kappa_r * theta_r) / (2 * delta_r)) + (kappa_r * i / 2))

            g_next = g[t + 1, i, j]
            g[t, i, j] = (A_ij * g_next +
                          C_ip1j * g[t + 1, i + 1, j] +
                          C_im1j * g[t + 1, i - 1, j] +
                          B_ijp1 * g[t + 1, i, j + 1] +
                          B_ijm1 * g[t + 1, i, j - 1])

            # Paiement de coupon tous les 6 mois
            if (1 < t < N_t) and abs(((t - 1) * delta_t) % 0.5) < 1e-8:
                g[t, i, j] += C

            # Valeur plafonnée
            g[t, i, j] = min(g[t, i, j], M)

    # Conditions aux bornes (r et s) à chaque t
    # Bords en r
    g[t, 0, :] = g[t, 1, :]
    g[t, N_r - 1, :] = g[t, N_r - 2, :]

    # Bords en s
    g[t, :, 0] = g[t, :, 1]
    g[t, :, N_s - 1] = g[t, :, N_s - 2]


i0 = round((r0 - r_min) / delta_r)+1 
j0 = round((s0 - s_min) / delta_s)+1  


D0 = g[0, i0, j0]
 
################################################################################
#5]

def find_yield(P, C, M, n, guess=0.05):
    def price_function(r):
        times = np.arange(0.5, 5.5, 0.5)  
        sum_coupons = np.sum(C * np.exp(-r * times))
        sum_principal = M * np.exp(-r * n)
        return sum_coupons + sum_principal - P

    
    sol = root_scalar(price_function, bracket=[-0.99, 1], method='brentq')

    return sol.root

    
ytm_solution = find_yield(D0, C, M, 5)

print(f"La valeur de la dette est: {D0:.2f}$")
print(f"Le taux de rendement est : {ytm_solution * 100:.2f}%")
