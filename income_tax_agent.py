# %%
from textwrap import dedent     # tap(들여쓰기)에 상관없이 쓸 수 있게.
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from typing import TypedDict
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# # 임베딩 설정

# %%
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# %% [markdown]
# # 벡터 db 로드

# %%
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name='chroma-tax',
    embedding_function=embedding,
    persist_directory='./chroma-tax'
)

# %% [markdown]
# # 리트리버 선언

# %%
retriever = vector_store.as_retriever(search_kwargs={'k':3})

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

# %%
def retrieve(state: AgentState) -> AgentState:
    """ 
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state
    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.        
    """
    query = state["query"]
    docs = retriever.invoke(query)

    return {**state, "context": docs}       # **state로 기존 state값을 보존 한 뒤 context만 뒤집어 씀.

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')

# %%
from langchain import hub

generate_prompt = hub.pull('rlm/rag-prompt')

def generate(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반에서 벡터 스토어에서 문사를 가져온다.
    Args:
        state (AgentState): 사용자의 질문한 내용을 포함한 에이전트의 현재 state
    Returns:
        AgentState: 생성된 응답을 포함하는 state를 포함합니다.
    """
    context = state['context']
    query = state['query']
    
    rag_chain = generate_prompt | llm       # 기본적으로 llm에게 질문할때는 str형태로 넣어야 하지만 chain으로 체이닝 하면 자동으로 문자열 반환해서 넣어줌.
    
    response = rag_chain.invoke({'question':query, 'context': context})
    
    return {**state, 'answer': response}
    

# %%
doc_releveance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevence(state: AgentState) -> Literal["relevant", "irrelevant"]:
    """
    주어진 state를 기반으로 문서의 관련성을 판단합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state
    Returns:
        Literal['relevant', 'irrelevant']: 문서가 관련성이 높으면 'relevant', 그렇지 않으면 'irrelevant'
    """
    
    query = state['query']
    context = state['context']
    
    doc_releveance_chain = doc_releveance_prompt | llm
    
    response = doc_releveance_chain.invoke({'question': query, 'documents': context})
    
    if response['Score'] == 1: return "relevant"
    return "irrelevant"

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리 사전을 참고해서 사용자의 질문을 변경해 주세요.                                              
사전: {dictionary}
질문: {{query}}
""")

def rewrite(state: AgentState) -> AgentState:
    """
    사용자의 질문을 사전을 고려하여 변경합니다.
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state
    Returns:
        AgentState: 변경된 질문을 포함하는 state를 반환합니다.
    """
    
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({'query':query})
    
    return {"query": response}    

# %% [markdown]
# # 할루시네이션 (거짓 정보 판단)
# ### self-rag 추가

# %%
from langchain_core.output_parsers import StrOutputParser
# 프롬프트 선언
hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")            # 질문에 대한 답변이 맘에 들지 않을때 프롬프트 변경하면 된다. 쿼리 변경 x
# llm의 답변을 엄격히 함.
hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

# %% [markdown]
# ### hallucination 여부 판단

# %%
def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    # hallucination 체이닝
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    
    response = hallucination_chain.invoke({'student_answer':answer, 'documents':context})
    
    print(f'응답: {response}')
    
    return response

# %% [markdown]
# # 답변의 품질 판단

# %%
from langchain import hub

helpfulness_prompt = hub.pull('langchain-ai/rag-answer-helpfulness')

# %% [markdown]
# ### 답변의 품질 판단 함수

# %%
def check_helpfulness_grader(state: AgentState) -> Literal['helpful', 'unhelpful']:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다.
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state
    Returns:
        Literal['helpful', 'unhelpful']: 답변이 유용하다고 판단되면 'helpful', 그렇지 않으면 'unhelpful'를 반환합니다.
    """
    # 데이터 로드
    query = state['query']
    answer = state['answer']
    # 체인
    helpfulness_chain = helpfulness_prompt | llm
    # 응답 객체
    response = helpfulness_chain.invoke({'question':query, 'student_answer': answer})
    
    if response['Score'] == 1 :
        print('유용성: helpful')
        return 'helpful'
    print('유용성: unhelpful')
    return 'unhelpful'

# %% [markdown]
# ### 그래프의 가시성을 위한 더미 노드 추가

# %%
def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수 입니다.
    """
    return state

# %%
builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("rewrite", rewrite)
builder.add_node("check_helpfulness", check_helpfulness)

# %%
builder.add_edge(START, "retrieve")
builder.add_conditional_edges(
    "retrieve", check_doc_relevence,
    {
        "relevant": "generate",
        "irrelevant": "rewrite"
    }
    )
builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)
builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
builder.add_edge("rewrite", "retrieve")
graph = builder.compile()