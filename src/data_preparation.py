
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_dummy_data(num_samples=100):
    data = {
        'text': [
            'Este documento fala sobre finanças e investimentos.',
            'Notícias sobre a nova política econômica do governo.',
            'Artigo científico sobre inteligência artificial e aprendizado de máquina.',
            'Receita de bolo de chocolate com cobertura de brigadeiro.',
            'Regulamentação de segurança para veículos autônomos.',
            'Relatório anual da empresa sobre lucros e perdas.',
            'Pesquisa sobre o impacto da tecnologia na educação.',
            'Guia completo para iniciantes em programação Python.',
            'Análise de mercado para o setor de energias renováveis.',
            'Contrato de aluguel de imóvel residencial.',
            'Discussão sobre os avanços da medicina e biotecnologia.',
            'Manual de instruções para montagem de móveis.',
            'Estudo sobre a psicologia do consumidor.',
            'Notícias sobre esportes e resultados de jogos.',
            'Documento legal sobre direitos autorais.',
            'Parecer jurídico sobre um caso de propriedade intelectual.',
            'Legislação ambiental para empresas de mineração.',
            'Termos de serviço de uma plataforma de software.',
            'Patente de um novo dispositivo eletrônico.',
            'Acordo de confidencialidade para startups de tecnologia.',
        ] * (num_samples // 20 + 1),
        'category': [
            'Finanças',
            'Política',
            'Tecnologia',
            'Culinária',
            'Tecnologia',
            'Finanças',
            'Educação',
            'Tecnologia',
            'Finanças',
            'Legal',
            'Saúde',
            'Casa',
            'Educação',
            'Esportes',
            'Legal',
            'Legal',
            'Legal',
            'Legal',
            'Tecnologia',
            'Legal',
        ] * (num_samples // 20 + 1)
    }
    df = pd.DataFrame(data)
    return df.sample(n=num_samples, random_state=42).reset_index(drop=True)


if __name__ == '__main__':
    df = generate_dummy_data(num_samples=200)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('../data/train.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)
    print('Dados de treino e teste gerados e salvos em data/train.csv e data/test.csv')


