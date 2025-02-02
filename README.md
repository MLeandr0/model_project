# Relatório do Teste Técnico Ford - Processamento de Linguagem Natural (NLP)

### Descrição da Abordagem Tomada
Foram utilizados dados de 2015 a 2019 com o objetivo de desenvolver um modelo capaz de classificar o principal ponto de problema nas reclamações. A estratégia adotada consistiu na utilização da coluna `COMPDESC` — que indica os componentes com problemas — para definir as classes de classificação, baseando-se nos componentes mais frequentes. Em seguida, o modelo foi treinado para classificar as reclamações a partir da coluna `CDESCR`, que contém a descrição textual das queixas.

![My Image](project_model.png)

## Etapas Tomadas

1. **Aquisição e Pré-processamento de Dados:**
Primeiramente, foi baixado o arquivo txt de dados e após uma conversão para csv para uma melhor vizualização, eu inspecionei o documento e percebi que as colunas não possuiam um header, por isso adicionei um de acordo com um dicionário presente na própria página do [NHTSA](https://static.nhtsa.gov/odi/ffdd/cmpl/Import_Instructions_Excel_All.pdf). Em seguida removi as colunas com uma maior quantidade de valores ausentes assim como colunas que eu não vi como seriam usadas no meu modelo. Além disso, nessa etapa, tratei os textos deixando todos minusculos, removendo números, espaços ausentes e também convertir valores de string para númericos, para assim poder usar mais facilmente depois. Por fim, removi as stop words em inglês presentes na coluna de reclamações.

2. **Engenharia de Features:**
A partir desse momento entrei em um território que eu tinha pouca experiência. Então após pesquisar muito, resolvi fazer uma vetorização TD-IDF como recomendado pela atividade, para converter o texto em uma matriz de características TF-IDF, limitando o número máximo de características para 10.000 e removendo stop words em inglês. Em seguida fiz o One-Hot para as principais features categoricas e uma normalização para o ano do problema. Implementei a classe TextMerger para combinar as colunas `CDESCR` e `COMPDESC` de texto relevantes em uma única entrada.

3. **Treinamento do Modelo:**
Criei uma pipeline integrando as etapas pré-processamento e classificação. Usei a regressão logística principalmente por conta da sua eficiente, pois todas as minhas outras ideias e testes demoravam demais. Em seguida, avaliei o modelo usando `accuracy_score` e gerei um relatório que no fim incluia precisão, recall e F1-score para cada classe criada. No fim, utilizei a biblioteca joblib para salvar o modelo treinado e retutilizar caso eu achasse necessário por algum motivo.

5. **Relatório e Análise Estatística:**

**Ponto Forte:**
- Acredito que um ponto forte foi a velocidade de treinamento em comparação com todas as minhas tentativas ao longo da semana. Como foi meu primeiro contato com um treinamento de um modelo, eu queria fazer algo mais simples porém funcional, uma vez que eu só tive experiência com tratamento e coleta dos dados.
- A utilização de pipelines que me permitiu facilmente recoletar os dados como também te-los facilmente em outros computadores.
- Os dados foram tratados removendo stopwords, convertendo valores númericos assim como a remoção das colunas que não iriam ser usadas.
- Fácil reutilização do modelo treinado.

**Ponto Fraco:**
- Criação das classes: A estratégia para definir as classes de treinamento poderia ter sido aprimorada, já que um número significativo de valores na coluna de origem das classes eram apenas `unknown`.
- Impacto de valores ausentes: A grande quantidade de dados ausentes em algumas colunas reduziu o leque de opções que poderiam ter sido usadas com os dados.
