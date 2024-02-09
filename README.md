# Comparativo entre métodos de construção de grafos para classificação semissupervisionada

Normalmente grandes bases de dados não vêm com todos seus dados rotulados de acordo com alguma característica que desejamos buscar, logo, trabalhar com aprendizado semissupervisionado pode ser interessante ao simular condições do mundo real.

Além disso, uma das formas de inputar os dados no aprendizado semissupervisionado é criando grafos a partir dos datasets fornecidos, tendo em vista que eles podem acelerar o desempenho computacional e gerar melhores resultados de predição de rótulos.

Uma das formas pensadas para poder criar grafos esparsos (característica desejada na criação de grafos para aprendizado semissupervisionado) é utilizando as primeiras etapas do algoritmo HDBSCAN*, um algoritmo de clusterização baseado em densidade, com os seguintes passos:

<p align="center">
  <img width="693" alt="image" src="https://github.com/rnlobao/graph-construction-SSL/assets/66230142/b49b753a-79c2-4a79-a667-639c0bed7d01">
</p>

O segundo passo gera uma árvore geradora mínima (grafo que conecta todos os pontos com menor custo de arestas):

<p align="center">
    <img width="400" alt="image" src="https://github.com/rnlobao/graph-construction-SSL/assets/66230142/2f706e60-90be-48b4-a248-e2694c73a1ad">
</p>

E esse grafo é esparso, o que abre brecha para testar se utilizando o HDBSCAN* podemos encontrar resultados melhores que o atual estado da arte.

O trabalho foi inspirado em um artigo publicado por Souza et al em 2013, em que utilizaram diversos métodos de construção de grafos e compararam seus resultados através do erro e derivação médio em diversos datasets, escolhidos por Chapelle em seu livro que é tido como uma bíblia do aprendizado semissupervisionado. O trabalho é composto pelas seguintes etapas:

<p align="center">
  <img width="700" alt="image" src="https://github.com/rnlobao/graph-construction-SSL/assets/66230142/a19c7672-7da7-4b68-906a-0a53e826a830">
</p>

E os seguintes resultados foram encontrados, sendo de 2 datasets melhor que do atual estado da arte:
<p align="center">
  <img width="700" alt="image" src="https://github.com/rnlobao/graph-construction-SSL/assets/66230142/4477410b-69fd-4c99-82a7-f9f13fd8a732">
</p>

Sendo meu trabalho de monografia defendido e aprovado com nota 9,7!

<p align="center">
  <img width="700" alt="Captura de Tela 2024-02-09 às 17 17 36" src="https://github.com/rnlobao/graph-construction-SSL/assets/66230142/c48d0f71-4d76-45c7-998e-d0be779e09d0">
</p>




