# Analisi del Sentiment - App di Machine Learning

Questa applicazione utilizza algoritmi di machine learning per analizzare automaticamente il sentiment di testi in **bahasa Indonesia**, classificandoli come positivi o negativi.

## 🚀 Caratteristiche Principali

- **Analisi Automatica del Sentiment**: Classifica testi come positivi o negativi
- **Lingua Supportata**: **Bahasa Indonesia** (indonesiano)
- **Preprocessing Avanzato**: Pulizia, stemming e normalizzazione del testo indonesiano
- **Machine Learning**: Utilizza Support Vector Machine (SVM) con TF-IDF
- **Validazione Incrociata**: Cross-validation 5-fold per risultati affidabili
- **Visualizzazione Risultati**: Grafici matplotlib per l'analisi dei dati

## 📊 Performance

L'applicazione raggiunge:
- **Accuracy**: 79%
- **Precision**: 76%
- **Recall**: 85%

## 🛠️ Requisiti di Sistema

- Python 3.7+
- Windows/Linux/macOS
- Processore moderno per l'addestramento del modello

## 📦 Installazione Completa

### Passo 1: Scaricare il Progetto
```bash
git clone https://github.com/ramaprakoso/analisis-sentimen.git
cd analisis-sentimen-1
```

### Passo 2: Creare l'Ambiente Virtuale
```bash
# Crea ambiente virtuale
python -m venv venv

# Attiva ambiente virtuale (Windows)
venv\Scripts\activate

# Attiva ambiente virtuale (Linux/macOS)
source venv/bin/activate
```

### Passo 3: Installare le Dipendenze
```bash
# Installa le librerie principali
pip install nltk numpy pandas scikit-learn matplotlib

# Verifica installazione
python -c "import nltk, numpy, pandas, sklearn; print('Tutte le dipendenze installate correttamente')"
```

### Passo 4: Scaricare i Dati NLTK (se necessario)
```bash
python -c "import nltk; nltk.download('punkt')"
```

## 🚀 Avvio dell'Applicazione

### Metodo 1: Avvio Rapido
```bash
# Assicurati di essere nella directory del progetto
cd "C:\Users\Computer\Documents\analisis-sentimen-1"

# Attiva l'ambiente virtuale (se non già attivo)
venv\Scripts\activate

# Avvia l'analisi del sentiment
python svm.py
```

### Metodo 2: Avvio con Ambiente Virtuale
```bash
# Attiva ambiente virtuale
venv\Scripts\activate

# Vai alla directory del progetto
cd "C:\Users\Computer\Documents\analisis-sentimen-1"

# Esegui l'applicazione
python svm.py
```

## 📁 Struttura del Progetto

```
analisis-sentimen-1/
├── svm.py                 # Script principale per l'analisi del sentiment
├── main.py               # Script di preprocessing dati
├── preprocessing.py      # Modulo per la pulizia del testo
├── README.md             # Questo file
├── dataset_final/        # Dataset di training e testing
│   ├── training90.csv    # 90% dati di addestramento
│   └── testing10.csv     # 10% dati di test
├── kamus/               # Dizionari e risorse linguistiche
│   ├── stopword.txt     # Parole da ignorare
│   ├── noise.txt        # Parole rumore
│   └── positif_ta.txt   # Parole positive
└── save_train/          # Modelli salvati
```

## 🔧 Configurazione

### Dataset
L'applicazione utilizza automaticamente:
- **Training set**: 90% dei dati per addestrare il modello
- **Test set**: 10% dei dati per valutare le performance

### Parametri del Modello
- **Algoritmo**: Support Vector Machine (SVM)
- **Kernel**: Lineare
- **Caratteristiche**: 2000 features TF-IDF
- **Cross-validation**: 5-fold

## 📊 Output Atteso

Quando esegui `python svm.py`, vedrai:
```
loading dictionary ...
Complate

Preparing data ...
Complate

Pipelining process ...
2000
Complate

classfication ...
Complate

Result ...
Recall :0.85
Precision :0.76
Accuracy :0.79
```

Seguito da un grafico a barre che mostra la distribuzione dei sentiment.

## 🐛 Risoluzione Problemi

### Errore: "Module not found"
```bash
# Assicurati che l'ambiente virtuale sia attivo
venv\Scripts\activate

# Reinstalla le dipendenze
pip install --upgrade nltk numpy pandas scikit-learn matplotlib
```

### Errore: "No module named 'sklearn'"
```bash
# Installa scikit-learn
pip install scikit-learn
```

### Warning sui caratteri Unicode
Questi sono solo avvisi e non influenzano il funzionamento dell'applicazione.

## 📈 Utilizzo Avanzato

### Personalizzare il Modello
Nel file `svm.py`, puoi modificare:
- Numero di caratteristiche: `max_features=2000`
- Split training/testing: `training90.csv`, `testing10.csv`
- Kernel SVM: `kernel='linear'`

### Aggiungere Nuovi Dati
1. Aggiungi i tuoi testi in formato CSV
2. Posizionali nella cartella `dataset_final/`
3. Aggiorna i percorsi nel codice se necessario

## 🤝 Contribuire

Sentiti libero di:
- Migliorare il preprocessing del testo
- Aggiungere nuovi algoritmi di classificazione
- Espandere il supporto linguistico
- Ottimizzare le performance

## 📝 Licenza

Questo progetto è open source e disponibile per uso educativo e di ricerca.

---

**Creato da**: Ramaprakoso
**Versione**: 1.0
**Linguaggio**: Python 3.7+
**Framework**: Scikit-learn, NLTK
