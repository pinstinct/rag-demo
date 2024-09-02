from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama


def simple(question: str):
    template = """Question: {question}
    Answer: Let's think step by step in Korean"""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOllama(model="llama3.1")
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"question": question}))


simple("컴퓨터에 대해 설명해줘.")
