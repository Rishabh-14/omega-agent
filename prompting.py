from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Prompt template and LLM
prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model

response = chain.invoke({"foo": "bears"})
print(response)

chain = prompt | model.bind(stop=["\n"])

#response = chain.invoke({"foo": "bears"})

#print(response)

# Function calling

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]
chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions)

#response = chain.invoke({"foo": "bears"}, config={})

#print(response)

# PromptTemplate + LLM + OutputParser


from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()

#response = chain.invoke({"foo": "bears"})

#print(response)

# Function output parser

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonOutputFunctionsParser()
)

#response = chain.invoke({"foo": "bears"})
#print(response)

#Simplifying input

from langchain_core.runnables import RunnableParallel, RunnablePassthrough


map_ = RunnableParallel(foo=RunnablePassthrough())
chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

#response = chain.invoke({"foo": "bears"})
#print(response)

chain = (
    {"foo": RunnablePassthrough()}
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

response = chain.invoke("bears")
print(response)