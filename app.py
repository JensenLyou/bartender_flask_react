from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask.helpers import send_from_directory
import agent
import os
from agent import Agent
from dotenv import load_dotenv

app = Flask(__name__, static_folder='./build', static_url_path='')
CORS(app)
# Load default environment variables (.env)
load_dotenv()

# AGENT_NAME = os.getenv("AGENT_NAME", "my-agent")
AGENT_NAME = "scrooge"

agent = Agent(AGENT_NAME)

# Creates Pinecone Index
agent.createIndex()

print("Talk to the AI!")
curUser = "User"
agent.setUser(curUser)


@app.route('/v1/chat', methods=['POST'])
@cross_origin()
def process_input():
    data = request.json
    user_input = data.get('user_input')

    if user_input:
        if user_input.startswith("read:"):
            agent.read(" ".join(user_input.split(" ")[1:]))
            response = {
                'result': 'Understood! The information is stored in my memory.'
            }
        elif user_input.startswith("newUser:"):
            cur_user = " ".join(user_input.split(" ")[1:])
            agent.setUser(cur_user)
            response = {
                'result': f'New user set to {cur_user}'
            }
        elif user_input == "clear":
            agent.clearMemory()
            response = {
                'result': 'Memory Cleared'
            }
        elif user_input == "toggleThoughts":
            agent.seeThoughts = not agent.seeThoughts
            response = {
                'result': 'Thoughts toggled, please enter a new message to see AI agent internal thoughts.'
            }
        elif user_input.startswith("viewMemory:"):
            username = " ".join(user_input.split(" ")[1:])
            content = agent.viewMemory(username)
            response = {
                'result': f'Memory viewed for user {username}\n{content}'
            }
        else:
            result = agent.action(user_input)
            response = {
                'result': result
            }
    else:
        response = {
            'error': 'Invalid input'
        }

    return jsonify(response)


@app.route('/v1/test', methods=['POST'])
@cross_origin()
def testPost():
    data = request.json

    result = {'message': 'Data received successfully!', 'data': data}
    return jsonify(result)


@app.route('/test', methods=['GET'])
@cross_origin()
def test():
    return 'Hello,This is test log'


@app.route('/')
@cross_origin()
def serve():
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT', 80))
