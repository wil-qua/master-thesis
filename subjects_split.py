import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Daten laden
df = pd.read_csv('HCP_YA_subjects_2026_10.csv')

print("Verteilung: Gesamter Datensatz")
print(df['Gender'].value_counts())
print(df['Age'].value_counts())
print("="*40)

MAIN_PATH = "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/"
NUM_TRAIN_PROBANDS = 60
NUM_VAL_PROBANDS = 12

# Hilfsfunktion für zu kleine Altersgruppen (vermeidet Abstürze)
def get_safe_stratify_labels(strat_series):
    counts = strat_series.value_counts()
    rare_classes = counts[counts < 2].index
    if len(rare_classes) > 0:
        majority_class = counts.idxmax()
        return strat_series.replace(rare_classes, majority_class)
    return strat_series

# 1. Datensatz nach Geschlecht trennen
df_f = df[df['Gender'] == 'F'].copy()
df_m = df[df['Gender'] == 'M'].copy()

# ==========================================
# FRAUEN (36 insgesamt -> 30 Train, 6 Val)
# ==========================================
# 36 Frauen aus allen Frauen ziehen (stratified nach Alter)
safe_age_f1 = get_safe_stratify_labels(df_f['Age'])
f_train_val, _ = train_test_split(df_f, train_size=36, stratify=safe_age_f1, random_state=42)

# Die 36 in 30 Train und 6 Val aufteilen
safe_age_f2 = get_safe_stratify_labels(f_train_val['Age'])
f_train, f_val = train_test_split(f_train_val, train_size=30, stratify=safe_age_f2, random_state=42)

# ==========================================
# MÄNNER (36 insgesamt -> 30 Train, 6 Val)
# ==========================================
# 36 Männer aus allen Männern ziehen (stratified nach Alter)
safe_age_m1 = get_safe_stratify_labels(df_m['Age'])
m_train_val, _ = train_test_split(df_m, train_size=36, stratify=safe_age_m1, random_state=42)

# Die 36 in 30 Train und 6 Val aufteilen
safe_age_m2 = get_safe_stratify_labels(m_train_val['Age'])
m_train, m_val = train_test_split(m_train_val, train_size=30, stratify=safe_age_m2, random_state=42)

# ==========================================
# 3. ZUSAMMENFÜHREN & MISCHEN
# ==========================================
# DataFrames zusammensetzen und durchmischen (sample frac=1 mischt die Zeilen durch)
df_train = pd.concat([f_train, m_train]).sample(frac=1, random_state=42).reset_index(drop=True)
df_val = pd.concat([f_val, m_val]).sample(frac=1, random_state=42).reset_index(drop=True)

# Ausgabe der finalen Verteilung zur Kontrolle
print("\n--- Prozentuale Verteilung Trainingsset (exakt 60) ---")
print(df_train[['Gender', 'Age']].value_counts(normalize=True).sort_index() * 100)

print("\n--- Prozentuale Verteilung Validierungsset (exakt 12) ---")
print(df_val[['Gender', 'Age']].value_counts(normalize=True).sort_index() * 100)

df_train['Subject'].to_csv('train_subjects.csv', index=False, header=False)
df_val['Subject'].to_csv('val_subjects.csv', index=False, header=False)

"""
all_subjects_in_dir = sorted(os.listdir(MAIN_PATH))
train_subjects = all_subjects_in_dir[:NUM_TRAIN_PROBANDS]
val_subjects = all_subjects_in_dir[NUM_TRAIN_PROBANDS : NUM_TRAIN_PROBANDS + NUM_VAL_PROBANDS]

def get_demographics_subset(subject_list, df):
    # Konvertiere IDs sicher in Strings zum Vergleichen
    sub_str_list = [str(s) for s in subject_list]
    df_copy = df.copy()
    df_copy['Subject_str'] = df_copy['Subject'].astype(str)
    return df_copy[df_copy['Subject_str'].isin(sub_str_list)]

train_df = get_demographics_subset(train_subjects, df)
val_df = get_demographics_subset(val_subjects, df)

print(f"Aktuelles TRAININGSSET ({NUM_TRAIN_PROBANDS} Probanden) - Verteilung:")
print("="*40)
print("Geschlecht:\n", train_df['Gender'].value_counts())
print("\nAlter:\n", train_df['Age'].value_counts())
print("\nKreuztabelle (Alter x Geschlecht):\n", pd.crosstab(train_df['Age'], train_df['Gender']))

print("\n" + "="*40)
print(f"Aktuelles VALIDIERUNGSSET ({NUM_VAL_PROBANDS} Probanden) - Verteilung:")
print("="*40)
print("Geschlecht:\n", val_df['Gender'].value_counts())
print("\nAlter:\n", val_df['Age'].value_counts())
print("\nKreuztabelle (Alter x Geschlecht):\n", pd.crosstab(val_df['Age'], val_df['Gender']))

# 2. Kombinierte Stratifizierungs-Spalte erstellen (Alter + Geschlecht)
df['Strat_Group'] = df['Age'].astype(str) + '_' + df['Gender']

# 3. Seltene Klassen (< 2 Proben) für den Stratifizierungsschritt anpassen
counts = df['Strat_Group'].value_counts()
rare_groups = counts[counts < 2].index

df['Strat_Group_Clean'] = df['Strat_Group']
for g in rare_groups:
    if '22-25' in g:
        df.loc[df['Strat_Group'] == g, 'Strat_Group_Clean'] = '22-25_M'

# 4. Erster Split (z. B. 80/20, um die Verhältnisse beizubehalten)
train_full, val_full = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['Strat_Group_Clean']
)

# 5. Exakte Stichprobengröße herunterskalieren (30 Train, 6 Val)
# Wir nutzen .sample(), um die stratifizierten Proportionen beizubehalten
train_df = train_full.sample(n=NUM_TRAIN_PROBANDS, random_state=42)
val_df = val_full.sample(n=NUM_VAL_PROBANDS, random_state=42)

# Aufräumen der Hilfsspalten
train_df = train_df.drop(columns=['Strat_Group', 'Strat_Group_Clean'])
val_df = val_df.drop(columns=['Strat_Group', 'Strat_Group_Clean'])

# 6. Listen der Dateipfade für dein config.py / training erstellen
# (Pass das an deine Spaltennamen an, z.B. wenn die Dateipfade aus den Subject-IDs generiert werden)
TRAIN_SUBJECTS = train_df['Subject'].tolist()
VAL_SUBJECTS = val_df['Subject'].tolist()

print(f"Ausgewählte Trainings-Probanden ({len(TRAIN_SUBJECTS)}): {TRAIN_SUBJECTS}")
print(f"Ausgewählte Validierungs-Probanden ({len(VAL_SUBJECTS)}): {VAL_SUBJECTS}")

print("\n" + "="*40)
print("TRAININGSSET (30 Probanden) - Verteilung:")
print("="*40)
print("Geschlecht:\n", train_df['Gender'].value_counts())
print("\nAlter:\n", train_df['Age'].value_counts())
print("\nKreuztabelle (Alter x Geschlecht):\n", pd.crosstab(train_df['Age'], train_df['Gender']))

print("\n" + "="*40)
print("VALIDIERUNGSSET (6 Probanden) - Verteilung:")
print("="*40)
print("Geschlecht:\n", val_df['Gender'].value_counts())
print("\nAlter:\n", val_df['Age'].value_counts())
print("\nKreuztabelle (Alter x Geschlecht):\n", pd.crosstab(val_df['Age'], val_df['Gender']))
"""