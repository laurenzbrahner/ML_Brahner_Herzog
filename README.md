# Insolvenzvorhersage polnischer Unternehmen mittels Machine Learning

**Modul:** Machine Learning  
**Autoren:** Laurenz Brahner, Maximilian Herzog

Binäre Klassifikation auf Basis von Jahresabschlussdaten des polnischen Markts (2000–2013). Verglichen werden Logistische Regression, Random Forest und XGBoost über fünf Prognosehorizonte (1–5 Jahre vor Insolvenz).

---

## Projektstruktur

```
├── data/               # ARFF-Rohdaten (1year.arff – 5year.arff)
├── models/             # Trainierte Modelle und Ergebnisse (.pkl)
├── notebooks/          # Jupyter Notebook (vollständige Analyse)
├── pages/              # Streamlit-Unterseiten
├── Info.py             # Streamlit-Einstiegspunkt
├── train_models.py     # Modelltraining (erzeugt models/)
└── requirements.txt    # Abhängigkeiten
```

---

## Voraussetzungen

```bash
pip install -r requirements.txt
```

---

## Notebook ausführen

```bash
jupyter notebook notebooks/bankruptcy_classification.ipynb
```

Das Notebook ist selbsterklärend und kann von oben nach unten durchgeführt werden. Alle Abhängigkeiten werden in der ersten Zelle per `%pip install` installiert.

---

## Streamlit-Anwendung starten

Die Modelle sind bereits trainiert und unter `models/` gespeichert. Die App kann direkt gestartet werden:

```bash
streamlit run Info.py
```

Die Anwendung öffnet sich automatisch im Browser unter `http://localhost:8501`.

Sollten die Modelldateien fehlen, müssen sie zunächst neu trainiert werden:

```bash
python train_models.py
streamlit run Info.py
```
