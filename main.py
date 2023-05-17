import agent
import os
from agent import Agent
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

#AGENT_NAME = os.getenv("AGENT_NAME", "my-agent")
AGENT_NAME = "Scrooge"

agent = Agent(AGENT_NAME)

# Creates Pinecone Index
agent.createIndex()

print("Talk to the AI!")
curUser = "User"
agent.setUser(curUser)

# Change user by typing "newUser: {insert name}"

while True:
    userInput = input()
    if userInput:
        if (userInput.startswith("read:")):
            agent.read(" ".join(userInput.split(" ")[1:]))
            print("Understood! The information is stored in my memory.")
        elif (userInput.startswith("newUser:")):
            curUser = " ".join(userInput.split(" ")[1:])
            agent.setUser(curUser)
            print(f"New user set to {curUser}")
        elif (userInput == "clear"):
            agent.clearMemory()
            print("Memory Cleared")
        elif (userInput == "toggleThoughts"):
            agent.seeThoughts = not agent.seeThoughts
        else:
            print(agent.action(userInput), "\n")
    else:
        print("SYSTEM - Give a valid input")
