import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI # LLM을 명시적으로 설정하기 위해 import
from llama_index.embeddings.openai import OpenAIEmbedding # 임베딩 모델을 명시적으로 설정

# .env 파일 로드
load_dotenv()

# API 키가 있는지 다시 한번 확인
if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OPENAI_API_KEY is not set. Please check your .env file.")

print("API 키 확인 완료.")

# LlamaIndex가 사용할 LLM과 임베딩 모델을 명시적으로 설정합니다.
# 이렇게 하면 어떤 모델을 쓰는지 명확히 알 수 있습니다.
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

try:
    print(" ./data 폴더에서 문서를 로딩합니다...")
    # 'data' 폴더에 있는 모든 문서를 재귀적으로 불러옵니다.
    documents = SimpleDirectoryReader("./data").load_data()
    
    if not documents:
        print("\n---[ 경고 ]---")
        print(" ./data 폴더에 읽을 수 있는 문서가 없습니다!")
        print("RAG를 테스트하려면 data 폴더에 PDF나 TXT 파일을 넣어주세요.")
        print("----------------\n")
    else:
        print(f"총 {len(documents)}개의 문서를 로딩했습니다.")

        print("문서 인덱싱을 시작합니다. (문서가 많으면 시간이 걸릴 수 있습니다...)")
        # 불러온 문서들로부터 벡터 인덱스(지식 창고)를 생성합니다.
        # 이 과정에서 내부적으로 OpenAI 임베딩 모델을 사용합니다.
        index = VectorStoreIndex.from_documents(documents)
        print("인덱싱 완료!")

        # 인덱스로부터 RAG 쿼리 엔진을 생성합니다.
        query_engine = index.as_query_engine()

        print("\n--- RAG 챗봇이 준비되었습니다 ---")
        print("data 폴더의 문서 내용을 기반으로 질문해보세요.")
        print("종료하려면 'exit'를 입력하세요.\n")

        # 사용자가 계속 질문할 수 있도록 반복문 실행
        while True:
            user_query = input("나: ")
            if user_query.lower() == 'exit':
                print("챗봇을 종료합니다.")
                break
            
            # 쿼리 엔진에 질문을 보냅니다.
            # 이 엔진이 알아서 문서를 검색하고(Retrieval),
            # 검색된 내용을 바탕으로 LLM에게 답변을 생성(Generation)시킵니다.
            response = query_engine.query(user_query)
            
            print(f"AI (RAG): {response}")

except Exception as e:
    print(f"\n---[ 오류 발생 ]---")
    print(f"오류 메시지: {e}")