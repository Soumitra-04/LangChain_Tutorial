import requests
from dotenv import load_dotenv #for loading API keys into thr environment

from langchain.agents import create_agent #to create agents to do stuff
from langchain.tools import tool #decorator to declare a function as a tool
from langchain_groq import ChatGroq
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
        {"role" : "user", "content" : "what is the weather like in Vienna?"}
    ] 
})

print(response)
print(response["messages"][-1].content)
