Repository: Previsione Multivariata dei Prezzi Elettrici (Day-Ahead Centro-Sud Italia)

Questa repository contiene i codici sorgenti elaborati nella Tesi sull'applicazione di architetture Deep Learning basate su LSTM ibride al forecasting energetico. Il focus primario è la comparazione dei framework per la previsione nel mercato MGP (Mercato del Giorno Prima) nel delicato assetto della zona Centro-Sud Italia, caratterizzata da un aumento di Fonti di Energia Rinnovabili (FER).

L'obiettivo dell'analisi è l'implementazione pratica del preprocessing (sequenze dinamiche a 72h e 168h e normalizzazione ad-hoc) e della logica architetturale per misurare metriche oggettive e definire il modello vincente.

Struttura della Repository

data/: Dataset storici.
src/: Moduli backend python.
architectures.py: Modelli (LSTM a 100 neuroni, Bi-LSTM a 140 unità, Convolutional-LSTM ibridi filtrati a 256 e Attention networks).
preprocessing.py: RobustScaler (immune agli outlier storici) operante su sliding window 3D (es: 72 osservazioni orarie).
evaluation.py: Computazione di MSE, MAE, RMSE e correttore analitico MAPE_safe mitigante il decadimento per denominatori nulli.
notebooks/: Runtime workflows di previsione e valutazione stagionale.
models_saved/ e results/: Storage automatizzato generato dal main script.
Tabella 12: Efficacia Architetturale (Indicatori MAE Finali)

I risultati del testing testimoniano come modelli puramente ricorrenti risultano letali nel brevissimo termine, contrariamente, orizzonti a vettore giornaliero dominano in topologie ibride tramite estrazione convoluzionale (CNN).

Stagione	Modello Orario Ottimale	MAE Orario	Modello Giornaliero Ottimale	MAE Giornaliero
Autunno	LSTM	6.605	CNN-LSTM-Bi	12.791
Inverno	Bi-LSTM	5.890	Bi-LSTM	11.607
Primavera	LSTM	7.496	CNN-LSTM-Bi	17.415
Estate	Bi-LSTM	5.641	CNN-LSTM-Bi	10.447
Le Verità Dietro Le Metriche (Overfitting & Cannibalizzazione)

1. Attenzione Inefficace

Benché teoricamente l'Attention offrisse massima flessibilità per serie storiche (cfr. architetture Z. Wang et al.), in test l'assunzione si è rivelata catastrofica, causando un grave overfitting. Il dispendio in complessità parametrica non è giustificato rispetto all'architettura standard Bi-LSTM che elabora bidirezionalmente lo stack orario temporale (MAE Invernale Insuperato = 5.890).

2. Il Paradosso della Primavera - "Cannibalizzazione Solare"

Analizzando l'anomala Primavera e le sue instabilità meteorologiche massicce tra i risultati, si nota un MAPE aberrante (che ha toccato massimi del 2345.105% nel forecasting giornaliero). La causale è interamente imputabile all'over-generation solare della zona Centro-Sud che abbatte drammaticamente il prezzo di mercato generando un effettivo output quasi nullo (€/Mwh ~ 0). Questa "Cannibalizzazione da Rinnovabile" manda in crash il calcolo d'errore percentuale. Da qui l'implementazione custom di MAPE_safe con approccio epsilon-offset in backend, assicurativamente introdotta in evaluation.py.

🤝 Contatti & Collaborazioni | Contact & Outreach

Se sei un recruiter, un ricercatore o un appassionato di Energy Forecasting e desideri approfondire l'analisi, non esitare a contattarmi.

Sono disponibile per:

Accesso ai Notebook Interattivi: Per visualizzare i log completi dell'addestramento e i grafici dinamici su Google Colab.
Discussione Tecnica: Approfondimenti sulla scelta delle iper-parametri e sulla gestione della volatilità del mercato CSUD.
Opportunità Professionali: Collaborazioni nell'ambito Data Science, Machine Learning ed Energy Analytics.
📩 Email: [Saccucciwork@gmail.com]
