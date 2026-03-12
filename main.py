import requests
from dotenv import load_dotenv #for loading API keys into thr environment

from langchain.agents import create_agent #to create agents to do stuff
from langchain.tools import tool,ToolRuntime #decorator to declare a function as a tool
from langchain_groq import ChatGroq
from pydantic import BaseModel

from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver 
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
@tool('get_weather', description = "Return weather information for a given city", return_direct= False)
def get_weather(city : str ):
    response = requests.get(f"https://wttr.in/{city}?format=j1") #return json
    return response.json()


model = ChatGroq(
    model="llama-3.3-70b-versatile"
)
#need to create this model becuase create_agent does not accept model_provider
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant who always cracks jokes and is humourous while remaining helpful"
)

response = agent.invoke({
    "messages" : [
        {"role" : "user", "content" : "what is the weather like in Mumbai?"}
    ]
})

# print(response)
print(response["messages"][-1].content)


model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature = 0.1
)

conversation = [
    SystemMessage("You are a helpful assistant for questions regarding programming"),
    HumanMessage("What is python and some common libraries used?"),
    AIMessage("Python is an interpreted language, common libraries used are NumPy, Pandas, MatplotLib, TensorFlow, etc."),
    HumanMessage("which language is better than python?")

]

#response = model.invoke(conversation)
# print(response)
# print(response.content)

# for chunk in model.stream("Hello what is python?"):
#     print(chunk.text, end = "",flush = True)


# while True:
#     q = input("You: ")

#     if q.lower() == "exit":
#         break

#     response = model.invoke(q)
#     print("AI:", response.content)


@dataclass
class Context:
    user_id : str


@dataclass
class ResponseFormat(BaseModel):
    summary : str
    temperature_fahrenheit: float
    temperature_celsius: float
    humidity : float

@tool('Locate_user' , description="Look up a user's city based on the context:")
def Locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id :
        case "ABC123":
            return "Vashi"
        case "XYZ456":
            return "Delhi"
        case "HJKL111":
            return "Satara" 
        case default:
            return "Unknown"
    
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools = [get_weather, Locate_user],
    system_prompt="You are a helpful weather assistant,who always cracks jokes while remaining helpful",
    context_schema= Context,
    response_format= ResponseFormat,
    checkpointer=checkpointer
)

# Config ={"configurable" : {"thread_id" : 1}}

# response = agent.invoke({
#     "messages" : [
#         {"role" : "user" , "content" : "What is the weather like?"}]},
#     config = Config,
#     context = Context(user_id="ABC123") 
# )
# print(response["structured_response"].summary)
# print("Temperature in Celsius: ",response["structured_response"].temperature_celsius)

message = {
    "role" : "user" ,
    "content" : [
        {"type" : "text", "text" : "Describe the contents of this image"},
        {"type" : "image", "url" : "https://www.vecteezy.com/free-photos/image"}
    ]
}

response = model.invoke([message])

print(response.content)