import pandas as pd
import numpy as np
import lightgbm as lgb
import pulp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# ==========================================
# CONFIGURĂRI (Aceleași ca în test.py)
# ==========================================
NUM_DAYS = 8
INTERVALS_PER_DAY = 96
TOTAL_INTERVALS = NUM_DAYS * INTERVALS_PER_DAY # 768
P_MAX = 10.0
C_MAX = 10.0
SOC_START = 5.0
MIN_TRANZACTIE = 0.1
LGBM_PARAMS = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 300,
    'learning_rate': 0.03, 'num_leaves': 128, 'max_depth': 8, 
    'n_jobs': -1, 'verbose': -1, 'seed': 42
}

# ==========================================
# 1. PREGĂTIREA DATELOR (SIMULARE)
# ==========================================
print(">>> 1. Încărcare și Split Date...")
df = pd.read_csv("Dataset.csv")

# Corecție format dată (pentru siguranță)
try:
    df['Start'] = pd.to_datetime(df['Time interval (CET/CEST)'].str.split(' - ').str[0], format="%d.%m.%Y %H:%M")
except:
    df['Start'] = pd.to_datetime(df['Time interval (CET/CEST)'].str.split(' - ').str[0], dayfirst=True)

# !!! AICI ESTE TRUCUL !!!
# Tăiem ultimele 8 zile din dataset pentru a le folosi ca VALIDARE
split_index = len(df) - TOTAL_INTERVALS

df_train = df.iloc[:split_index].copy()  # Trecutul (ce vede modelul)
df_truth = df.iloc[split_index:].copy().reset_index(drop=True) # Viitorul real (ce ascundem)

print(f"   Antrenăm pe: {len(df_train)} intervale")
print(f"   Testăm pe:   {len(df_truth)} intervale (Perioada: {df_truth['Start'].iloc[0]} -> {df_truth['Start'].iloc[-1]})")

# Feature Engineering
def create_features(df_in):
    d = df_in.copy()
    d['hour'] = d['Start'].dt.hour
    d['minute'] = d['Start'].dt.minute
    d['dow'] = d['Start'].dt.dayofweek
    d['is_weekend'] = (d['dow'] >= 5).astype(int)
    # Lags
    if 'Price' in d.columns:
        d['lag_96'] = d['Price'].shift(96)
        d['lag_48'] = d['Price'].shift(48)
        d['lag_672'] = d['Price'].shift(672)
    return d.dropna().reset_index(drop=True)

df_train_proc = create_features(df_train)
features = ['hour', 'minute', 'dow', 'is_weekend', 'lag_96', 'lag_48', 'lag_672']

# ==========================================
# 2. ANTRENARE MODEL
# ==========================================
print("\n>>> 2. Antrenare LightGBM...")
model = lgb.LGBMRegressor(**LGBM_PARAMS)
model.fit(df_train_proc[features], df_train_proc['Price'])
# ==========================================
# 3. PREDICȚIE RECURSIVĂ (SIMULARE CORECTATĂ)
# ==========================================
print("\n>>> 3. Predicție Recursivă (Simulation)...")

# 1. Inițializăm istoricul cu datele de antrenament
history = df_train['Price'].tolist() 
predicted_prices = []

# 2. Pregătim DataFrame-ul de bază pentru predicție (DOAR caracteristicile de timp)
# IMPORTANT: Nu folosim funcția create_features cu dropna aici, facem manual!
pred_df = df_truth.copy()
pred_df['hour'] = pred_df['Start'].dt.hour
pred_df['minute'] = pred_df['Start'].dt.minute
pred_df['dow'] = pred_df['Start'].dt.dayofweek
pred_df['is_weekend'] = (pred_df['dow'] >= 5).astype(int)

# Ne asigurăm că Features sunt în ordinea corectă (fără coloanele de lag încă)
time_features = ['hour', 'minute', 'dow', 'is_weekend']
feature_order = ['hour', 'minute', 'dow', 'is_weekend', 'lag_96', 'lag_48', 'lag_672']

print(f"   Vom prezice {len(pred_df)} intervale (Trebuie să fie 768)...")

# 3. Bucla de predicție
for i in tqdm(range(len(pred_df))):
    # Calculăm lungimea curentă a istoricului pentru a găsi indicii corecți
    curr_len = len(history)
    
    # Extragem valorile de lag din istoric (care crește dinamic)
    lag_96_val = history[curr_len - 96]
    lag_48_val = history[curr_len - 48]
    lag_672_val = history[curr_len - 672]
    
    # Construim un mic DataFrame pentru O SINGURĂ linie (fixează warning-urile)
    row_dict = {
        'hour': [pred_df.iloc[i]['hour']],
        'minute': [pred_df.iloc[i]['minute']],
        'dow': [pred_df.iloc[i]['dow']],
        'is_weekend': [pred_df.iloc[i]['is_weekend']],
        'lag_96': [lag_96_val],
        'lag_48': [lag_48_val],
        'lag_672': [lag_672_val]
    }
    
    X_single_row = pd.DataFrame(row_dict)
    
    # Asigurăm ordinea coloanelor ca la antrenare
    X_single_row = X_single_row[feature_order]
    
    # Prezicem
    pred = model.predict(X_single_row)[0]
    
    predicted_prices.append(pred)
    history.append(pred) # Adăugăm predicția în istoric pentru pasul următor!

# Verificare lungimi
print(f"   Lungime Predicții: {len(predicted_prices)}")
print(f"   Lungime Adevăr:    {len(df_truth)}")

# Calcul eroare
mae = mean_absolute_error(df_truth['Price'], predicted_prices)
print(f"   MAE Preț: {mae:.2f}")
# ==========================================
# 4. OPTIMIZARE MILP
# ==========================================
print("\n>>> 4. Rezolvare MILP...")

def solve_milp(prices):
    prob = pulp.LpProblem("Test_MILP", pulp.LpMaximize)
    T = 96
    pos = pulp.LpVariable.dicts("pos", range(T), -P_MAX, P_MAX)
    soc = pulp.LpVariable.dicts("soc", range(T+1), 0, C_MAX)
    chg = pulp.LpVariable.dicts("chg", range(T), cat='Binary')
    dis = pulp.LpVariable.dicts("dis", range(T), cat='Binary')
    surplus = pulp.LpVariable("surplus", 0, C_MAX)

    # Obiectiv: Profit Tranzacții + Surplus * MinPrice
    # Profit = -1 * pos * price (pt că pos>0 e cumpărare, deci cost)
    # Dar stai! În codul tău original: (discharge * price) - (charge * price)
    # Asta e echivalent cu: -1 * position * price.
    
    min_price = min(prices)
    obj = pulp.lpSum([-1 * pos[t] * prices[t] for t in range(T)]) + (surplus * min_price)
    prob += obj

    prob += soc[0] == SOC_START
    for t in range(T):
        prob += soc[t+1] == soc[t] + pos[t]
        prob += chg[t] + dis[t] == 1
        prob += pos[t] >= MIN_TRANZACTIE * chg[t] - P_MAX * dis[t]
        prob += pos[t] <= P_MAX * chg[t] - MIN_TRANZACTIE * dis[t]
    
    prob += soc[T] == surplus # Surplusul e ce rămâne la final

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return [pulp.value(pos[t]) for t in range(T)]

final_actions = []
for d in range(NUM_DAYS):
    p_day = predicted_prices[d*96 : (d+1)*96]
    final_actions.extend(solve_milp(p_day))

# ==========================================
# 5. CALCUL PROFIT REAL
# ==========================================
real_prices = df_truth['Price'].values
total_profit = 0

for i in range(len(final_actions)):
    # Profit = -1 * Acțiune * Preț_REAL
    # (Dacă vând (-10), fac -1 * -10 * Price = +Profit)
    # (Dacă cumpăr (+10), fac -1 * 10 * Price = -Cost)
    revenue = -1 * final_actions[i] * real_prices[i]
    total_profit += revenue

# Adăugăm surplusul de la finalul fiecărei zile (dacă există)
# Notă: MILP-ul tău e setat să vândă surplusul la prețul minim prezis.
# În calculul real, el se vinde la prețul minim REAL.
for d in range(NUM_DAYS):
    idx_start = d * 96
    idx_end = (d+1) * 96
    day_acts = final_actions[idx_start:idx_end]
    day_real_prices = real_prices[idx_start:idx_end]
    
    # Calculăm soc final
    soc = 5.0
    for a in day_acts:
        soc += a
    
    if soc > 0.01:
        sale_val = soc * min(day_real_prices)
        total_profit += sale_val

print(f"\n=============================================")
print(f"REZULTAT FINAL PE ULTIMELE 8 ZILE CUNOSCUTE:")
print(f"PROFIT ESTIMAT: {total_profit:,.2f} EUR")
print(f"=============================================")

# Plotare
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Preț Real', alpha=0.5)
plt.plot(predicted_prices, label='Preț Prezice', alpha=0.7)
# Scalăm acțiunile să se vadă pe grafic (ex: x10)
plt.plot([a*10 for a in final_actions], label='Acțiuni (x10)', alpha=0.3)
plt.legend()
plt.title("Backtest: Real vs Prezicere vs Acțiuni")
plt.show()