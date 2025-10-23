# KAI - Kalkyle AI

Recover is Scandinavia’s leading company in damage control and restoration.
In Norway, we work with all the major insurance companies and carry out around 30,000 assignments each year—mainly insurance-related damages such as water and fire damage.
We have over 1,200 employees in Norway, most of whom are skilled tradespeople such as carpenters and painters. We also have a significant number of remediation specialists who are experts in disinfection, cleaning, and odor removal after incidents involving soot, mold, chemical spills, and blood. The term *“sanering”* (remediation) is one you might come across during the hackathon, it simply means cleaning, decontaminating, or restoring something to a healthy condition.

When damage occurs, we respond on short notice and take immediate action to limit the damage. We then document the site with photos and reports, and calculate the full extent of the damage in detail. Once the insurance company approves the estimate, we begin the work of cleaning, drying, demolishing, and rebuilding so that the site is restored to its original condition.

Recover usually acts as the total provider for all post-damage work. This means we “take care of everything needed” and invoice the insurance company. If plumbers or electricians are required, we hire them for the specific damage case.


**All information about the case is within the `docs/Hackathon_Recover.pdf` file.**
**Submit your solutions on the [Kaggle competition page](https://www.kaggle.com/competitions/hackathon-recover-x-cogito/overview).**

## 🛠️ Prerequisites

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

1. Clone this repository

    ```bash
    git clone https://github.com/CogitoNTNU/recover-hackathon.git
    cd recover-hackathon
    ```

2. Retrieve data On Kaggle
    - Create a Kaggle account if you don't have one.
    - Navigate to the [Recover X Cogito Hackathon](https://www.kaggle.com/competitions/hackathon-recover-x-cogito/overview) page.
    
    Either use Kaggle API or download the data manually:
    - **Using Kaggle API**: [Kaggle API Documentation](https://www.kaggle.com/discussions/getting-started/524433) 
      1. Go to the 'Account' tab on your Kaggle profile.
      2. Click 'Create New Token'. This will download a file named kaggle.json containing your API credentials.
      3. Move this file to the appropriate location:
         * Linux/OSX: ~/.kaggle/kaggle.json
           * chmod 600 ~/.kaggle/kaggle.json
         * Windows: C:\Users\<Windows-username>\.kaggle\kaggle.json
         Make sure the permissions are set correctly to keep the file secure.
      4. ```bash
            uv run python dataset/hackathon.py
            ```
    - **Download the dataset manually**:
      1. Go to the [Data](https://www.kaggle.com/competitions/hackathon-recover-x-cogito/data) page.
      2. Download the dataset files.
      3. Extract the downloaded files into the `data/` directory of this repository.

3. Install dependencies:
    One can either use `uv` or `pip` to install the dependencies.
    Using `uv`:
    ```bash
    uv sync
    ```
    Using `pip`:
    ```bash
    pip install -r requirements.txt
    ```


## Data

The dataset used in this project is a custom dataset that contains information about calculations in
insurance claims. The dataset is stored in the `data/` folder.

### Tabular Data

### Image Data

### Text Data



# Oversikt
Dette prosjektet inneholder datasett og hjelpefunksjoner for å trene modeller på **arbeidsoperasjoner i rom** innen Recover sine prosjekter.  
Målet er å forutsi hvilke arbeidsoperasjoner som mangler i et gitt rom, basert på både rommet selv og konteksten fra resten av prosjektet.

Kjernen er **HackathonDataset**, som kombinerer:

- **WorkOperationsDataset** – arbeidsoperasjoner i rom (X, Y, calculus, room_cluster)
- **ProjectsDataset** – prosjektmetadata (forsikringsselskap, postnummer, distanse mellom kontor og skadested m.m.)

Hvert datapunkt inneholder dermed både **romnivå-data** og **prosjektnivå-data**.

---

## Datastruktur

### Felter fra WorkOperationsDataset
- `X` – synlige arbeidsoperasjoner i rommet (one-hot vektor)
- `Y` – skjulte arbeidsoperasjoner som skal predikeres (one-hot vektor)
- `X_codes` – synlige arbeidsoperasjoner som liste med koder
- `Y_codes` – skjulte arbeidsoperasjoner som liste med koder
- `room_cluster` – standardisert romkategori (f.eks. `bad`, `kjøkken`, `ukjent`)
- `room_cluster_one_hot` – one-hot representasjon av romkategori
- `calculus` – kontekst fra andre rom i prosjektet (liste av strukturer med arbeidsoperasjoner + romkategori)

### Felter fra ProjectsDataset
- `project_id` – unik identifikator for prosjektet
- `insurance_company` – forsikringsselskap
- `insurance_company_one_hot` – one-hot representasjon av selskap
- `recover_office_zip_code`, `damage_address_zip_code` – postnumre
- `office_distance` – distanse mellom kontor og skadested
- `case_creation_year`, `case_creation_month` – tidspunkt for skadesaken

---

## Sampling strategy
Datasettet simulerer manglende operasjoner ved å skjule et tilfeldig utvalg arbeidsoperasjoner i hvert rom.  
Dette styres av en **sampling_strategy** – en liste med konfigurasjoner, for eksempel:

```python
[
  {"subset_size": 0.5, "sample_pct": 0.5, "use_weighted_sampling": True, "use_sampled_calculus": True},
  {"subset_size": 0.5, "sample_pct": 0.3, "use_weighted_sampling": False, "use_sampled_calculus": True},
]
```

- `subset_size` → hvor stor andel av datasettet strategien gjelder (må summere til 1.0)
- `sample_pct` → hvor stor andel av arbeidsoperasjonene i et rom som kan skjules
- `use_weighted_sampling` → vektlegger sjeldne arbeidsoperasjoner (basert på `tickets.csv`)
- `use_sampled_calculus` → bestemmer om calculus skal bruke skjulte eller originale operasjoner

\
Datasettet reshuffles med `dataset.shuffle()`, som trekker nye operasjoner å skjule.\
Ved å benytte deg av metoden `dataset.set_sampling_strategy(sampling_strategy)` før shuffle, kan du veksle \
mellom vanskelige og enkle sampling strategier og antall permutasjoner i datasettet kan dermed bli veldig stort.

### Seed
Du kan angi `seed` ved opprettelse av HackathonDataset eller WorkOperationsDataset.  
Dette gir deterministisk sampling (samme splitting mellom X og Y).

```python
dataset = HackathonDataset(split="train", download=True, sas_token=sas_token, seed=42)
```

---

## Evaluering (score.py)
For evaluering brukes en tilpasset rom-score med vekting av riktige og gale prediksjoner.

- **True Positive (TP)** = +1  
- **False Positive (FP)** = –0.25  
- **False Negative (FN)** = –0.5  
- **Ingen targets og ingen prediksjoner i et rom** = +1  

Funksjoner:
```python
from metrics import get_room_scores, normalized_rooms_score

# Score for ett rom
room_scores = get_room_scores(preds, targets)

# Score over alle rom
score = normalized_rooms_score(all_preds, all_targets)
```

---

## Collate-funksjon (collate.py)
For batch-trening i PyTorch kan du ta utgangspunkt i `collate_fn`, som pakker sammen features og kontekst:

```python
from dataset import collate_fn
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

Output fra `collate_fn`:
- `X` – features for rommet (arbeidsoperasjoner + romkategori)
- `Y` – labels (de skjulte operasjonene)
- `context` – tensor med andre rom i prosjektet
- `context_mask` – maskering av gyldige kontekst-rom

---

## Pandas og Polars
Hvis du ikke ønsker å bruke PyTorch eller batch loading, kan du hente datasettet som **tabeller**.  
Dette gir et litt annet blikk på dataene og er nyttig for utforsking eller klassisk dataanalyse.

```python
from dataset import HackathonDataset

dataset = HackathonDataset(split="train", download=True, sas_token=sas_token)

df_polars = dataset.get_polars_dataframe()   # Polars DataFrame
df_pandas = dataset.get_pandas_dataframe()   # Pandas DataFrame
```

Her blir `WorkOperationsDataset` koblet med `ProjectsDataset` via `project_id`, slik at du får ett samlet datasett.

Vi anbefaler [polars](https://www.pola.rs/) på det varmeste, med det er selvfølgelig ingenting i veien med pandas dersom dere er mer komfortable der (men det går tregere).

---

## Eksempelbruk

```python
from dataset import HackathonDataset, collate_fn
from metrics.score import normalized_rooms_score
from torch.utils.data import DataLoader

# Last inn treningssett
dataset = HackathonDataset(split="train", download=True, seed=42, root="data").get_pandas_dataframe()

# Se på en rad
sample = dataset[0]
print(sample["X"])        # synlige operasjoner (one-hot)
print(sample["Y"])        # skjulte operasjoner (one-hot)
print(sample["X_codes"])  # synlige koder
print(dataset.to_cluster_names(sample["X_codes"]))  # dekode til lesbare navn

# Bruk med DataLoader
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
batch = next(iter(loader))

# Eller hent som tabell
df = dataset.get_polars_dataframe()
print(df.head())


# Scoring av prediksjoner
preds = [
    [3, 49],      # Modellens prediksjoner i rom 1
    [12],         # Modellens prediksjoner i rom 2
    [],           # Modellens prediksjoner i rom 3 (ingen forslag)
]

targets = [
    [3, 49, 129], # Fasiten for rom 1
    [12, 77],     # Fasiten for rom 2
    [],           # Fasiten for rom 3 (ingen operasjoner mangler)
]

score = normalized_rooms_score(preds, targets)
print("Normalized score:", score)

```

---

## Oppsummering
- **HackathonDataset** er inngangspunktet – den kombinerer romdata og prosjektdata.  
- Du kan bruke datasettet i **PyTorch** med batch loading og collate.  
- Eller du kan hente det som **Polars/Pandas** for enkel analyse.  
- Sampling-strategi og seed gir kontroll over vanskelighetsgrad og reproduserbarhet.  
- Evaluering skjer med en tilpasset **rom-score** som balanserer TP, FP, FN og tomme rom. 
 