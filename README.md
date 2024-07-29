Primo scheletro del progetto di rilevamento audio deepfake.
Cartella training (da rinominare): circa 11k file audio, divisi in fake e real. Tutti utilizzati per popolare il json dei dati. Usati anche per l'apprendimento, divisi in training, validation e test.
main.py primo abbozzo di progetto, ancora nessun effort di refactoring o simili. Per ora un solo modello addestrato, una rete convoluzionale

Possibili passi successivi: 
fase preliminare di data augmentation e/o grafici di esempio delle feature usate (mfcc / mel spectrogram);
creazione e training di modelli diversi ispirandosi ai vari paper sull'argomento (LCNN/LSTM/RNN/GMM/RawNet);
ricerca di possibili sintomi di overfitting/underfitting
tuning degli iperparametri quali learning rate/numero di epoche di apprendimento
salvataggio grafici di train/validation loss&accuracy dei vari modelli e al variare dei modelli, degli iperparametri e di altri fattori quali presenza o meno di regolarizzazione
