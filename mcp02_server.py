from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma(
    collection_name='chroma-tax',
    embedding_function=embedding,
    persist_directory="./chroma-tax"
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

llm = ChatOpenAI(model="gpt-4o", streaming=True)
small_llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

mcp = FastMCP("Demo21")

@mcp.tool(
    name='retrieve_tool',
    description="""
    질문에 대해 관련 문서를 검색하여 반환하는 도구입니다.
    """
)

def retrieve_tool(query: str) -> str:
    """질문에 대해 관련 문서를 검색하여 반환하는 도구입니다."""
    docs = retriever.invoke(query)
    
    return format_docs(docs)

@mcp.tool(
    name='generate_tool',
    description="""검색된 문서(context)와 질문을 바탕으로 응답을 생성하는 도구입니다.""",
)

def generate_tool(context: str, question: str) -> str:
    
    prompt = PromptTemplate.from_template("""
    다음은 질문에 대한 문서입니다. 문서를 참고하여 질문에 대한 정확한 답변을 생성해 주세요. 
    
    [문서]
    {context}
    
    [질문]
    {question}                                     
    """)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({'context': context, 'question': question})

dictionary = ['사람과 관련된 표현 -> 거주자']
    
rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
사전: {dictionary}                                              
질문: {{query}}
""")

@mcp.tool(
    name='rewrite_tool',
    description= """
    사용자의 질문을 사전을 고려하여 변경합니다.     
    """,
)

def rewrite_tool(query: str) -> str:
    """
    사용자의 질문을 사전을 고려하여 변경합니다.     
    """
   
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()    
    
    return rewrite_chain.invoke({'query': query})

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":    
    mcp.run(transport='stdio')
    
