# Lista de empresas situadas na Russia

![bandeiras-russia-e-ucrania](https://github.com/rnlobao/Faculdade/assets/66230142/6fdd633a-a3c7-4c52-b362-49cfe33d2538)


Projeto de mineraçao de dados a partir do dataset disponibilizado no kaggle sobre a permanencia ou saida de determinadas empresas da Russia, de acordo com sua situaçao atual de guerra contra a Ucrania.

Contextualização:
Os dados do projeto foram obtidos do Kaggle, do dataset "List of Companies Leaving or Staying in Russia".

## 1. Problema
Tendo em vista que vivemos em um mundo globalizado e a Russia na história moderna foi um dos ultimos países a abrir suas fronteiras para empresas multinacionais, o recente conflito armado contra a Ucrânia repercurtiu muito nessas empresas fazendo-las tomar atitudes de permanecer ou não em território Russo.


### 1.2 Objetivo
Identificar, dada uma multinacional situada na russia com seu país sede e tipo de indústria, identificar se ela ficaria ou não na Russia em um cenário pós-guerra.

Além disso, as seguintes questões devem ser respondidas:

- Empresas sediadas em países da OTAN tem mais chance de ficar?
- Como se comportaram as empresas que são sediadas nos EUA?
- Qual setor de negócio tem mais chance de sair do país?

### 3. Ferramentas e Processos
Quais ferramentas serão usadas no processo?
- Python 3.8.0;
- Jupyter Notebook;
- Git e Github;
- Pandas, Sklearn;

### 3.2 Processo
Fez-se um pré-processamento de dados utilizando:
- Para os atributos nominais e ordinais, fez-se a distribuição de probabilidade de cada
- Alteração do atributo categórico ordinal Grade de A, B, C, D, E, F para números
- Identificação dos valores aberrantes e inconsistências.
- Identificação de Outliers (usando Z score) e valores nulos
- Preenchimento de valores ausentes
- Resolução de inconsistências
- Transformação de atributos categóricos nominais em um vetor numérico

## 4. Os 3 principais insights dos dados
Feita a geração de regras de associação geramos os seguintes insights:
![Unknown](https://github.com/rnlobao/Russia-Companies/assets/66230142/910d38c8-7170-4f27-a52d-0ac5190ce602)
![Unknown-2](https://github.com/rnlobao/Russia-Companies/assets/66230142/95d7b20e-7f2c-466b-831a-086c181e6e17)
![Unknown-3](https://github.com/rnlobao/Russia-Companies/assets/66230142/1cdcbcc2-9481-4ee7-bfbc-0897e3d16c9e)


Tendo em vista que buscamos traçar o perfil de empresas que tendem a sair da russia, e as regras geradas, podemos ver o seguinte:
Empresas sediadas nos Estados Unidos, de cunho industrial e que participam da OTAN são mais tendenciosas a sair.

# 5. Modelos de Machine Learning aplicados
Foram aplicados 4 modelos de Machine Learning: SVM, Modelo de Regressão Logística, Random Forest e KNN

# 6. Performance do modelo de Machine Learning
Acurácia encontrada:
- SVM: 0.675
- Regressão Logística: 0.700
- Random Forest: 0.685
- KNN: 0.685

![image](https://github.com/rnlobao/Russia-Companies/assets/66230142/e1fef040-2ae5-485e-8c51-c3fb5a1f9d8f)


F1-Score:
- SVM: 0.176
- Regressão Logística: 0.000
- Random Forest: 0.383
- KNN: 0.479

![image](https://github.com/rnlobao/Russia-Companies/assets/66230142/b85c2f64-5662-434b-a1f6-92f147bb36af)


AUC-ROC:
- SVM: 0.515
- Regressão Logística: 0.500
- Random Forest: 0.582
- KNN: 0.627

![image](https://github.com/rnlobao/Russia-Companies/assets/66230142/132fa646-9c32-4dbc-988a-961f0e2f2e96)


# 7. Conclusões
Tendo em vista que nosso dataset é pequeno e não há diferenciação entre test e train foram obtidos bons resultados com o pré-processamento tendo altas chances de fazer uma boa predição utilizando Knn, já que em regressão logística o AUC-ROC = 0.5 diz que modelo não tem poder de discriminação e está apenas fazendo previsões aleatórias.

O trabalho foi realizado junto com Rafael (https://github.com/rafaugusto20) e Pablo (https://github.com/PabloCoellho)

## 8 Referências
* O Dataset foi obtido no [Kaggle](https://www.kaggle.com/datasets/vadimtynchenko/list-of-companies-leaving-or-staying-in-russia).
