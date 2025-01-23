필요 라이브러리 설치
!pip install pymongo
!pip install langchain_community
​
사용 모듈 불러오기
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import requests
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
import os
​
MongoDB 접속정보
MONGO_URI = "mongodb+srv://[계정명]:[비밀번호]@cluster0.gey2c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = [데이터베이스명]
COLLECTION_NAME = [컬렉션명]
​
OpenAI API 접속정보
AZURE_OPENAI_ENDPOINT = "https://[서브도메인명].openai.azure.com"
AZURE_API_KEY = [API키]
EMBEDDING_MODEL = "jc-intra-text-embedding-3-small"
TRANSLATION_MODEL = "jc-intra-gpt-4o"
os.environ["OPENAI_API_VERSION"] = "2024-06-01"
​
TMMCC 변수
# 하이브리드 검색 가중치 설정
BM25_MAX_VALUE = 10.0  # 설정 필요
BM25_MIN_VALUE = 0.0   # 설정 필요
VECTOR_SCORE_WEIGHT = 0.5
TEXT_SCORE_WEIGHT = 0.5
​
몽고DB 클라이언트
# MongoDB 클라이언트 생성
client = MongoClient(MONGO_URI)
db: Database = client[DATABASE_NAME]
collection: Collection = db[COLLECTION_NAME]
​
벡터 검색 함수
def vector_search(query_vector, vector_index_name, num_candidates=64, limit=25):
    """
    벡터 검색 수행
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit
            }
        },
        {
            "$project": {
                # "metadata": 1,
                "content": 1,
                # "media": 1,
                "vectorScore": {"$meta": "vectorSearchScore"},
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$sort": {"score": -1}
        },
        {
            "$limit": limit
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)
​
텍스트 검색 함수
def text_search(query, text_index_name, limit=25):
    """
    텍스트 검색 수행
    """
    pipeline = [
        {
            "$search": {
                "index": text_index_name,
                "text": {
                    "query": query,
                    "path": ["content", "metadata.KO", "metadata.ENG"]
                }
            }
        },
        {
            "$project": {
                # "metadata": 1,
                "content": 1,
                # "media": 1,
                "textScore": {"$meta": "searchScore"},
                "score": {"$meta": "searchScore"}
            }
        },
        {
            "$sort": {"score": -1}
        },
        {
            "$limit": limit
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)
​
임베딩 함수
def get_embedding_from_azure(text):
    """
    Azure OpenAI API를 사용해 텍스트 임베딩 생성
    """
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_MODEL}/embeddings?api-version=2023-05-15"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY,
    }
    data = {"input": text}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Azure OpenAI API 호출 실패: {response.text}")

    embedding = response.json()["data"][0]["embedding"]
    return embedding
​
TMMCC 계산을 위한 함수
arXiv.orgAn Analysis of Fusion Functions for Hybrid Retrieval​
def normalize_vector_score(vector_score):
    return (vector_score + 1) / 2.0

def normalize_bm25_score(bm25_score):
    return min((bm25_score - BM25_MIN_VALUE) / (BM25_MAX_VALUE - BM25_MIN_VALUE), 1.0)
    
def calculate_convex_score(vector_score, bm25_score):
    tmm_vector_score = normalize_vector_score(vector_score)
    tmm_bm25_score = normalize_bm25_score(bm25_score)
    return VECTOR_SCORE_WEIGHT * tmm_vector_score + TEXT_SCORE_WEIGHT * tmm_bm25_score
​
Hybrid Search
def hybrid_search(query):
    # 벡터 및 텍스트 검색 수행
    embedding = get_embedding_from_azure(query)
    vector_results = vector_search(embedding, "word_vector_index")
    text_results = text_search(query,"word_text_index")

    print(len(vector_results))
    print(len(text_results))
    # 결과 병합
    combined_results = {}
    for result in vector_results:
        doc_id = result["_id"]
        vector_score = result.get("vectorScore", 0)
        combined_results[doc_id] = {
            **result,
            "vectorScore": vector_score,
            "score": calculate_convex_score(vector_score, 0)
        }

    for result in text_results:
        doc_id = result["_id"]
        text_score = result.get("textScore", 0)
        if doc_id not in combined_results:
            combined_results[doc_id] = {
                **result,
                "vectorScore": 0,
                "score": calculate_convex_score(0, text_score)
            }
        else:
            vector_score = combined_results[doc_id]["vectorScore"]
            combined_results[doc_id]["textScore"] = text_score
            combined_results[doc_id]["score"] = calculate_convex_score(vector_score, text_score)

    # 결과 정렬
    sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results
​
hybrid 검색 예시
query = "재미있는 동굴로 가자! 상상수를 구해야해!"
type_filter = "report"
results = hybrid_search(query)
for result in results:
    print(result)
​
metadata를 array로 반환
def create_metadata_array(query, limit=25):
    """
    Hybrid Search를 수행하여 metadata를 array 형태로 묶은 JSON string 반환
    """
    # Hybrid Search 수행
    search_results = hybrid_search(query)
    
    # metadata array 생성
    metadata_array = []
    for result in search_results[:limit]:
        metadata = result.get("metadata", {})
        # metadata의 key-value를 배열 형태로 변환
        # metadata_entry = [{"key": key, "value": value} for key, value in metadata.items()]
        
        # 배열 추가
        metadata_array.append(metadata)

    # metadata array를 JSON string으로 변환
    metadata_json = json.dumps(metadata_array, ensure_ascii=False)

    return metadata_json
​
엑셀 용어사전 작성
def process_excel_with_glossary(input_file, output_file):
    """
    엑셀 파일의 source 데이터를 기반으로 Hybrid Search 결과를 glossary 컬럼에 추가
    """
    # 엑셀 파일 읽기
    df = pd.read_excel(input_file, dtype=str)
    df = df.fillna("(None)")
    df = df.replace([np.inf, -np.inf], 0)  # Infinity를 0으로 대체
    # glossary 컬럼 추가
    glossary_data = []
    for _, row in df.iterrows():
        source_text = row["OriTextData"]  # source 컬럼 값
        
        # Hybrid Search 수행
        search_results = hybrid_search(source_text)
        
        # metadata array 생성
        metadata_array = [result.get("metadata", {}) for result in search_results]
        
        # JSON string으로 변환
        metadata_json = json.dumps(metadata_array, ensure_ascii=False)
        
        # glossary 데이터 추가
        glossary_data.append(metadata_json)

    # 새로운 컬럼 추가
    df["glossary"] = glossary_data

    # 결과 저장
    df.to_excel(output_file, index=False)
    print(f"처리된 데이터가 엑셀 파일로 저장되었습니다: {output_file}")
​
Langchain을 이용한 번역
번역 프롬프트 템플릿
# 번역 프롬프트 템플릿
translation_prompt_template = """
You are a professional translator. Translate the given query into {target_language}.
Use the provided glossary to maintain consistency in terminology.

Glossary:
KO	ENG	RU	ES-LA	PT-PT	TUR	FR	GER	TH	VN
{glossary}

Query to translate:
{query}

Translation:
"""
​
Langchain LLM 모델 생성
# Azure OpenAI 모델 초기화
llm = AzureChatOpenAI(
    deployment_name=TRANSLATION_MODEL,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_API_KEY,
    temperature=0.5,
    max_tokens=512
)
​
Langchain Prompt Tempate 생성
prompt_template = PromptTemplate(
    input_variables=["query", "target_language", "glossary"],
    template=translation_prompt_template
)
​
Langchain 생성
# LLMChain 설정
translation_chain = LLMChain(llm=llm, prompt=prompt_template)
​
용어사전 만드는 함수
def create_glossary_from_results(results):
    """
    검색 결과를 기반으로 용어 사전 생성
    :param results: 검색 결과 리스트
    :return: 용어 사전 문자열
    """
    glossary = []
    for result in results:
        content = result.get("content", "")
        glossary.append(content.strip())
    return "\n".join(glossary)
​
번역 함수
def translate_query_with_glossary(query, results, target_language="English"):
    """
    검색 결과를 용어 사전으로 활용해 유저 쿼리를 번역
    :param query: 유저 쿼리
    :param results: 검색 결과 리스트
    :param target_language: 번역할 언어
    :return: 번역된 쿼리
    """
    glossary = create_glossary_from_results(results)
    translation = translation_chain.run({
        "query": query,
        "target_language": target_language,
        "glossary": glossary
    })
    return translation.strip()
​
query = "재미있는 동굴로 가자! 상상수를 구해야해!"
type_filter = "report"
results = hybrid_search(query)

 # 번역 수행
translated_query = translate_query_with_glossary(query, results, "English")
​
Original Query: 재미있는 동굴로 가자! 상상수를 구해야해!
Translated Query: Let's go to the Funny Cave! We need to save the Tree of Imagination!
