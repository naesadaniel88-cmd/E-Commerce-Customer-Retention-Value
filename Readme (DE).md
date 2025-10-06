# Olist Brasilianischer E‑Commerce — Kundenbindung & Wert (Churn‑Modellierung)

*Übersetzung der bereitgestellten README-Datei.*

**TL;DR** — Mit dem öffentlichen **Olist Brazilian E‑Commerce**‑Datensatz messen wir Churn über eine **Inaktivitätsregel (6 Monate)**, erstellen kundenbezogene Features (Frequency, Monetary, Sentiment) und liefern ein **Baseline‑Churn‑Modell** (Random Forest) mit **ROC‑AUC ≈ 0,70**. Das Repository umfasst **EDA**, **Churn‑Sensitivität** sowie **umsetzbare Retention‑Playbooks** auf Basis von Recency und Wert.

**Technik**: Python (Pandas, NumPy, SciPy, scikit‑learn) • Seaborn/Matplotlib  
**Daten**: Olist Brazilian E‑Commerce Public Dataset (orders, customers, items, payments, reviews, products, sellers, categories)  
**Repo‑Struktur**: [`notebooks/`](notebooks) • [`data/`](data) • [`assets/`](assets)

---

## Inhaltsverzeichnis
- [Hintergrund](#hintergrund)
- [Datensatz](#datensatz)
- [Methodik](#methodik)
- [Wichtigste Erkenntnisse](#wichtigste-erkenntnisse)
- [So reproduzierst du es](#so-reproduzierst-du-es)
- [Artefakte](#artefakte)
- [Business‑Empfehlungen](#business-empfehlungen)

---

## Hintergrund
E‑Commerce‑Teams fehlt oft ein klarer, wiederholbarer Weg, **Churn zu messen** und **Retention zu priorisieren**. Dieses Projekt verwandelt den Olist‑Datensatz in **Signale auf Kundenebene** (Frequency, Monetary, Sentiment) und einen **Baseline‑Classifier**, um Hochrisiko‑Accounts zu identifizieren. Die Pipeline bleibt bewusst einfach, reproduzierbar und business‑tauglich: Churn sauber definieren, **Label‑Leakage** vermeiden und die Modellgüte mit transparenten Metriken berichten.

---
### Geschäftsfrage
Welche Faktoren sagen **Kundenabwanderung** und **Wiederkauf** am besten voraus – und wie übersetzen wir diese Signale in **Retention‑Maßnahmen** mit dem höchsten ROI?

---

## Datensatz
- **Quelle**: Öffentliches [Olist Brazilian E‑Commerce‑Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (8 CSV‑Dateien, ~120 MB gesamt).  
- **Kernentitäten** (im Notebook geladen):  
  - `orders`, `customers`, `order_items`, `order_payments`, `order_reviews`, `products`, `sellers`, `product_category_name_translation`.
- **Datumsfelder & Parsing**: Alle Zeitstempel werden zu `datetime64` geparst – für verlässliche Zeitfilter und Kohorten.  
- **Kundenlabel (Churn)**: Ein Kunde gilt als **abgewandert**, wenn sein **letztes Kaufdatum** **vor `max(order_purchase_timestamp) − 6 Monate`** liegt.  
  - Sensitivitätsanalyse: 3‑, 6‑, 9‑ und 12‑Monats‑Cutoffs.

> **Hinweis**: „**Recency**“ (Tage seit letztem Kauf) ist hervorragend für die *Analyse*, wird aber aus den **Trainingsfeatures ausgeschlossen**, um Leakage zu vermeiden (es ist am Max‑Datum des Datensatzes verankert).

---

## Methodik

### 1) Datenladen & Qualitätskontrolle
- Alle 8 Tabellen lesen; Timestamps konvertieren; Shapes/Dtypes/Nulls prüfen; grundlegende Sanity‑Checks.  
- Keys zusammenführen und **Sichten auf Kundenebene** erstellen.

### 2) Churn‑Definition & Sensitivität
- `last_purchase` je Kunde berechnen und Inaktivitäts‑Cutoff anwenden (Standard **6 Monate**).  
- **Gesamt‑Churnrate** berichten und **Sensitivitätskurve** über 3/6/9/12 Monate zeigen.

### 3) Feature Engineering (Kundenebene)
- **Frequency**: `order_count` (Anzahl Bestellungen).  
- **Monetary**: je Kunde summiertes `payment_value` (`monetary`).  
- **Sentiment/Experience**: mittlerer `review_score` je Kunde (`avg_review`).  
- **Nur EDA** (nicht im Modell): `recency_days` seit letztem Kauf.

### 4) Statistische Tests (aktiv vs. abgewandert)
- **Mann‑Whitney‑U**‑Tests für Unterschiede in `order_count`, `monetary` und `avg_review` zwischen aktiven und abgewanderten Gruppen.

### 5) Modellierung
- **Baseline‑Features**: `["order_count", "monetary", "avg_review"]`.  
- **Train/Test‑Split**: 70/30, stratifiziert nach Label.  
- **Modelle**:  
  - **Logistische Regression** (`class_weight="balanced"`, `max_iter=1000`) — linearer Baseline.  
  - **Random Forest** (`n_estimators=200`, `random_state=42`, `class_weight="balanced"`) — nichtlinearer Baseline.  
- **Metriken**: ROC‑AUC (primär) plus Classification Report auf dem Testset.  
- **Erklärbarkeit**: **Feature Importances** des Random Forest (erwartet dominieren Monetary & Frequency; Review liefert Zusatzsignal).

### 6) Visualisierung
- Churn nach Stadt (mit Support‑Schwelle).  
- Churn‑Sensitivität vs. Cutoff‑Fenster.  
- Korrelations‑Heatmap (engineerte Features vs. Churn).  
- Verteilungen nach Label (Frequency, Ausgaben, Reviews).

---

## Wichtigste Erkenntnisse

### Aus dem Notebook (EDA & Modellierung)
- **Hoher Churn bei 6‑Monats‑Inaktivität**: **≈ 70,3 %** der Kunden gelten als abgewandert (gemäß Definition).  
- **Recency ist der stärkste Rohindikator**: **Korrelation ≈ +0,72** mit Churn — jeder zusätzliche Tag ohne Kauf erhöht das Risiko (nur EDA, nicht fürs Training).  
- **Monetary zählt am meisten**: Im **Random Forest** trägt der Spend die größte Bedeutung; High‑Value‑Kunden verhalten sich beim Churn anders.  
- **Frequency & Sentiment helfen**: Mehr Bestellungen und höhere durchschnittliche Reviews korrelieren mit Retention; Mann‑Whitney zeigt, dass Churner **signifikant weniger** bestellen (*p* ≈ 0,0000).  
- **Modell‑Performance (Baseline)**:  
  - **Random Forest**: **ROC‑AUC ≈ 0,70** (guter Startwert fürs Targeting).  
  - **Logistische Regression**: **ROC‑AUC ≈ 0,54** (nichtlineare Muster sind wichtig).

> Diese Scores sind bewusst **Baseline** (ohne starkes Tuning/Embeddings). Sie eignen sich, um Kunden nach Risiko zu **ranken** und **schnelle Retention‑Maßnahmen** zu starten.

---

## So reproduzierst du es

1) **Repository klonen.**  

2) **Daten‑Setup**  
   **Quelle**: Öffentliches [Olist Brazilian E‑Commerce‑Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (8 CSV‑Dateien, ~120 MB gesamt).  
   - **Dateien**:  
     `olist_orders_dataset.csv`, `olist_customers_dataset.csv`, `olist_order_items_dataset.csv`,  
     `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`,  
     `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `product_category_name_translation.csv`.  
   - **Lokaler Pfad (für Notebook erforderlich)**:  
     Lade alle CSVs von Kaggle und lege sie lokal ab, z. B.:  
     ```python
     DATA_DIR = r"D:\Brazilian E-Commerce Public Dataset by Olist\"
     ```
     Stelle sicher, dass dieser Ordner **alle acht CSVs** enthält, bevor du das Notebook ausführst.

3) **Umgebung einrichten**  
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   # source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Artefakte
- Notebook: [Brazilian E‑Commerce Public Dataset by Olist.ipynb](./Brazilian%20E-Commerce%20Public%20Dataset%20by%20Olist.ipynb)

---

## Business‑Empfehlungen (überprüft & datengetrieben)

### 1) Auf eine Churn‑Definition einigen, die zu eurem Kaufzyklus passt
**Beobachtung:** Mit **6‑Monats‑Inaktivität** liegt die **Churn‑Rate bei ~70,3 %**. Sensitivität: **3 Mon.: 89,6 %**, **6 Mon.: 70,3 %**, **9 Mon.: 49,1 %**, **12 Mon.: 29,7 %** (siehe „Churn Sensitivity“).  
**Warum wichtig:** Der Cutoff verändert KPI und Größe der „Save“‑Zielgruppe drastisch.  
**Aktion:**  
- Typischer Wiederkaufzyklus **≤ 6 Monate** → **6‑Monats‑Definition** beibehalten.  
- Langsam drehende Güter → **9–12 Monate**, um normale Pausen nicht als „Churn“ zu labeln.

---

### 2) Recency als primären **operativen** Trigger nutzen (nicht als Trainingsfeature)
**Beobachtung:** `recency_days` korreliert in der EDA stark mit Churn (**~ +0,72**); Recency wurde korrekt vom Training ausgeschlossen (Leakage‑Vermeidung).  
**Warum wichtig:** Recency ist das beste **Frühwarnsignal** für den Betrieb, auch wenn es fürs Training ungeeignet ist.  
**Aktion:**  
- **Recency‑Leiter** mit Triggern bei **60/90/120/150/180 Tagen** seit letztem Kauf aufbauen; automatische Aktionen/Kampagnen konfigurieren — z. B. **Erinnerung bei 60 Tagen**, **Rabatt bei 120 Tagen**, **stärkeres Win‑Back bei 180 Tagen**.  
- Je näher am Churn, desto **aggressiver** re‑engagen (größere Angebote, persönliche Nachrichten, exklusive Deals) — letzte Chance zum Zurückholen.

---

### 3) High‑Value‑Kunden priorisieren (Monetary treibt das Modell)
**Beobachtung:** Im Random Forest hat **Monetary ≈ 0,99** Importance; **Order Count ≈ 0,00**; **Avg. Review ≈ 0,01**. Richtung: mehr Ausgaben und mehr Orders → höhere Retention (Mann‑Whitney: alle *p* ≈ 0,0000).  
**Warum wichtig:** **Wertsegmente** machen Retention‑Budget effizienter.  
**Aktion:**  
- **Gestaffelte Playbooks nach Wert** — z. B. Top‑20 % erhalten Concierge‑Support, Prioritätsversand, exklusive Angebote; niedrigere Tiers automatisierte Angebote.  
- **Inkrementelle Marge** je Tier tracken, damit „Save“‑Kosten wirtschaftlich sind (insb. über Tiers wie Platinum/Gold/Silver).

---

### 4) Reviews als „Friction“‑Signal nutzen — auch wenn im Modell schwächer
**Beobachtung:** Das Modell stützt sich vor allem auf **Monetary (Gesamtausgaben)**; `Avg. Review` trägt weniger.  
**Warum wichtig:** Reviews bleiben **verlässliche Indikatoren** für Experience‑Probleme.  
**Aktion:**  
- Bei **niedrigen Bewertungen (1★–2★)** automatisch erkennen und **Service‑Recovery** auslösen, bevor Churn eintritt.

---

### 5) Modell zum **Risiko‑Ranking** einsetzen; nicht auf „Accuracy“ fixieren
**Beobachtung:** Random Forest **ROC‑AUC ≈ 0,704** > Logistische **≈ 0,543**. Reportete Accuracies (RF ≈ 0,70, LR ≈ 0,49) sind bei unausgeglichenen Labels ungeeignet.  
**Warum wichtig:** AUC zeigt, dass das Modell **Priorisierung** ermöglicht – nicht bloß binäre Klassifikation.  
**Aktion:**  
- Kunden **wöchentlich scoren** und die **Top 10–20 % Risiko** für Win‑Back ansprechen.  
- Schwellenwerte nach **Kosten‑Nutzen** wählen (Rabattkosten vs. erwarteter Save‑Wert), nicht nach roher Accuracy.  
- Später **Wahrscheinlichkeiten kalibrieren** (budgetbegrenztes Targeting). So könnt ihr Kunden nach **tatsächlicher Churn‑Wahrscheinlichkeit** sortieren und z. B. ab **P(Churn) ≥ 0,65** exakt so viele targeten, wie Budget & ROI erlauben.

---

### 6) Kontrollierte Experimente fahren und **Uplift** messen (nicht Klicks)
**Beobachtung:** Klare Trennung aktiv vs. abgewandert (signifikante Tests) + moderate AUC ⇒ Targeting hilft, **Business‑Lift** muss aber belegt werden.  
**Warum wichtig:** Nur **inkrementeller Uplift** validiert ROI.  
**Aktion:**  
- **A/B‑Tests**: Modell‑Zielgruppe vs. Control auf **Reaktivierung (30/60 Tage)**, **inkrementellen Umsatz** und **Marge**. So zeigt ihr Wirkung **außerhalb** des Notebooks.  
- **Reporting nach Wert‑Tier** (wo wirkt Budget am besten?).  
- **Inkrementelle Marge** tracken: Wenn ihr Geld (Rabatte/Ads/Outreach) zum „Retten“ eines Churn‑Risiko‑Kunden ausgebt, prüft, dass der **zusätzliche Gewinn** die Kosten übersteigt.
