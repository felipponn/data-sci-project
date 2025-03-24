import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Se você nunca baixou esses recursos NLTK, descomente as linhas abaixo:
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('rslp')
# nltk.download('punkt_tab')

def clean_texts(queries_dict, docs_dict):
    """
    Função que limpa e preprocessa os textos dos dicionários de queries e documentos
    
    Args:
        queries_dict: dicionário com {query_id: texto da query}
        docs_dict: dicionário com {doc_id: texto do documento}
        
    Returns:
        tuple: (queries limpas, documentos limpos)
    """
    stopwords_pt = set(stopwords.words('english'))
    stemmer = RSLPStemmer()
    
    def preprocess_text(text):
        # 1. Converte para minúsculas
        text = text.lower()
        # 2. Remove pontuação
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        # 3. Tokeniza
        tokens = nltk.word_tokenize(text, language='english')
        # 4. Remove stopwords e tokens que não são alfabéticos
        tokens = [t for t in tokens if t.isalpha() and t not in stopwords_pt]
        # 5. Aplica stemming
        tokens = [stemmer.stem(t) for t in tokens]
        # 6. Retorna texto "limpo"
        return ' '.join(tokens)

    # Criando versões limpas dos dicionários de queries e docs
    queries_dict_clean = {qid: preprocess_text(qtext) for qid, qtext in queries_dict.items()}
    docs_dict_clean = {did: preprocess_text(dtext) for did, dtext in docs_dict.items()}
    
    return queries_dict_clean, docs_dict_clean