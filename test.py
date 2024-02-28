from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Prompt template and LLM
prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model

response = chain.invoke({"foo": "bears"})
print(response)
