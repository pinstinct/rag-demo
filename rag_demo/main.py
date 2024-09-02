import hashlib

import bs4
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class SimpleRag:
    def __init__(self, url, question):
        self.url = url
        self.question = question
        self.embeddings = OllamaEmbeddings(model="llama3.1")
        self.store = chromadb.PersistentClient("../chroma_data")
        self.collection = self.store.get_or_create_collection(name="rag_demo")
        self.chat = ChatOllama(model="llama3.1")

    def get_data(self):
        loader = WebBaseLoader(web_paths=(self.url,),
                               bs_kwargs=dict(
                                   parse_only=bs4.SoupStrainer("div", attrs={"class": ["mw-body-content"]}, )))
        docs = loader.load()
        return docs

    def get_chunk(self):
        data = self.get_data()
        splitter = SemanticChunker(self.embeddings)
        chunks = splitter.split_documents(data)
        return chunks

    def generate_ids(self, chunk_content):
        url_hash = hashlib.sha256(self.url.encode()).hexdigest()
        chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
        return f"{url_hash}-{chunk_hash}"

    def save_store(self):
        chunks = self.get_chunk()
        documents = [doc.page_content for doc in chunks]
        ids = [self.generate_ids(doc.page_content) for doc in chunks]
        self.collection.upsert(ids=ids, documents=documents)

    def get_query(self):
        self.embeddings.embed_query(self.question)

    def result(self):
        self.save_store()
        context = self.collection.query(query_texts=self.question, n_results=1)
        template = """당신은 질문-답변(Question-Answer) Task 를 수행하는 AI 어시스턴트 입니다.
        검색된 문맥(context)를 사용하여 질문(question)에 답하세요.
        만약, 문맥(context)으로부터 답을 찾을 수 없다면 '모른다'고 말하세요.
        question: {question}, context: {context}
        한국어로 대답하세요."""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.chat | StrOutputParser()
        print(chain.invoke({"question": self.question, "context": context}))


rag = SimpleRag(
    "https://ko.wikipedia.org/wiki/%EA%B5%90%EB%B3%B4%EC%83%9D%EB%AA%85%EB%B3%B4%ED%97%98",
    "교보생명 빌딩에 있는 대사관은 어떤게 있어?"
)
print(rag.result())
