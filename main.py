from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
#response = llm.invoke("how can langsmith help with testing?")
#print(response.content)

#Prompt template

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
response = chain.invoke({"input": "how can langsmith help with testing?"})
print(response)
''''
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com")

docs = loader.load()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
'''