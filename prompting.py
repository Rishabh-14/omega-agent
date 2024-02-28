from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI(openai_api_key="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")
chain = prompt | model

response = chain.invoke({"foo": "bears"})
print(response)

chain = prompt | model.bind(stop=["\n"])

response = chain.invoke({"foo": "bears"})

print(response)