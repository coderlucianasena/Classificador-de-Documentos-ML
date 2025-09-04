# Projeto de Classificação de Documentos com Machine Learning

Este projeto implementa um classificador de documentos utilizando Machine Learning (Naive Bayes) e uma interface web simples com Flask.

## 🚀 Estrutura do Projeto

```
classificacao_documentos_ml/
├── data/                 # Dados de treino e teste
├── models/               # Modelos de ML treinados (vetorizador e classificador)
├── src/                  # Código fonte da aplicação
│   ├── data_preparation.py  # Script para gerar e preparar os dados
│   ├── model_training.py    # Script para treinar o modelo de ML
│   ├── model_evaluation.py  # Script para avaliar o modelo de ML
│   ├── predict.py           # Script para realizar previsões com o modelo
│   ├── app.py               # Aplicação web Flask
│   ├── test_model.py        # Testes unitários para o modelo
│   └── templates/           # Templates HTML para a aplicação Flask
│       └── index.html
├── docs/                 # Documentação adicional (se houver)
├── requirements.txt      # Dependências do projeto
└── README.md             # Este arquivo
```

## 🚀 Fases do projeto
```
## Tarefas do Projeto de Classificação de Documentos

### Fase 1: Planejamento e estrutura do projeto
- [x] Criar a estrutura de diretórios do projeto.
- [x] Definir as tecnologias e bibliotecas a serem utilizadas.

### Fase 2: Implementação do modelo de ML
- [x] Preparar os dados para treinamento e teste.
- [x] Implementar o modelo de classificação (ex: Naive Bayes, SVM, Redes Neurais).
- [x] Treinar e avaliar o modelo.

### Fase 3: Criação de dados de exemplo e testes
- [x] Gerar dados de exemplo para demonstração.
- [x] Escrever testes unitários para o modelo.

### Fase 4: Interface e documentação
- [x] Desenvolver uma interface simples para interação (ex: Flask, Streamlit).
- [x] Documentar o código e o uso do projeto.
```

## 🛠️ Configuração e Instalação

1.  **Clone o repositório (se aplicável):**

    ```bash
    git clone https://github.com/coderlucianasena/Classificador-de-Documentos-ML.git
    cd classificacao_documentos_ml
    ```

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

## 📖 Como Usar

### 1. Preparar os Dados

Execute o script de preparação de dados para gerar os arquivos `train.csv` e `test.csv` na pasta `data/`.

```bash
python src/data_preparation.py
```

### 2. Treinar o Modelo

Após a preparação dos dados, treine o modelo de classificação. Isso salvará o vetorizador TF-IDF e o modelo Naive Bayes na pasta `models/`.

```bash
python src/model_training.py
```

### 3. Avaliar o Modelo (Opcional)

Para verificar a performance do modelo, execute o script de avaliação:

```bash
python src/model_evaluation.py
```

### 4. Executar a Aplicação Web

Inicie a aplicação Flask. Ela estará disponível em `http://127.0.0.1:5000`.

```bash
python src/app.py
```

Abra seu navegador e acesse o endereço para usar a interface de classificação.

### 5. Classificar Documentos via Script

Você também pode usar o script `predict.py` para classificar documentos diretamente via linha de comando:

```bash
python src/predict.py
```

Este script contém exemplos de uso para classificar novos textos.

### 6. API REST

O projeto também oferece uma API REST para classificação de documentos:

**Endpoint**: `POST /classify`

**Exemplo de uso com curl**:
```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a document about finance and investments."}'
```

**Resposta**:
```json
{
  "text": "This is a document about finance and investments.",
  "category": "Finanças"
}
```

## Testes

Para executar os testes unitários do modelo, utilize:

```bash
python -m unittest src.test_model -v
```

## 📊 Performance do Modelo

O modelo treinado apresenta excelente performance:

- **Precisão**: 100%
- **Recall**: 100%
- **F1-Score**: 100%
- **Acurácia**: 100%

### Categorias Suportadas
- Finanças
- Política
- Tecnologia
- Culinária
- Educação
- Esportes
- Casa
- Saúde
- Legal

## 🔧 Solução de Problemas

### Erro de Caminho de Arquivo
Se você encontrar erros como "FileNotFoundError" ou "Cannot save file into a non-existent directory", certifique-se de executar os comandos do diretório raiz do projeto, não de dentro da pasta `src/`.

### Problemas de Encoding
Para textos em português, pode haver problemas de encoding. Use textos em inglês para testes ou configure o encoding UTF-8 adequadamente.

### Porta 5000 Ocupada
Se a porta 5000 estiver ocupada, você pode alterar a porta no arquivo `src/app.py`:
```python
app.run(debug=True, host="0.0.0.0", port=5001)  # Mude para 5001 ou outra porta
```

## 🛠️ Tecnologias Utilizadas

-   **Python**
-   **scikit-learn**: Para Machine Learning (TF-IDF, Naive Bayes).
-   **pandas**: Para manipulação de dados.
-   **Flask**: Para a criação da interface web.
-   **joblib**: Para salvar e carregar modelos de ML.

## 📝 Licença

Este projeto foi criado como exemplo educacional e pode ser modificado conforme necessário.

## 👩🏻‍💻 Criado por:

- Luciana Sena 
- 2025
