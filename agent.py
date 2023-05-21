import openai
import os
import pinecone
import yaml
from dotenv import load_dotenv
import nltk
from langchain.text_splitter import NLTKTextSplitter
import json

# Download NLTK for Reading
nltk.download('punkt')

# Initialize Text Splitter
text_splitter = NLTKTextSplitter(chunk_size=2500)

# Load default environment variables (.env)
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
# PINECONE_API_ENV = "asia-southeast1-gcp"

# Prompt Initialization
with open('prompts.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# Counter Initialization
with open('memory_count.yaml', 'r') as f:
    counter = yaml.load(f, Loader=yaml.FullLoader)

# Thought types, used in Pinecone Namespace
THOUGHTS = "Thoughts"
QUERIES = "Queries"
INFORMATION = "Information"
ACTIONS = "Actions"

# Top matches length
k_n = 3

# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# initialize openAI
openai.api_key = "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"


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def read_txtFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def generate(prompt):

    messages = []

    messages.append({'role': 'system', 'content': data['identity_prompt']})
    messages.append({'role': 'system', 'content': data['backstory_prompt']})
    messages.append({'role': 'system', 'content': data['world_prompt']})
    messages.append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages
    )

    return completion.choices[0].message["content"]


class Agent():
    def __init__(self, table_name=None) -> None:
        self.table_name = table_name
        self.memory = None
        self.thought_id_count = int(counter['count'])
        self.curUser = ""
        self.chat_indexes = []
        self.seeThoughts = False

    def set_init_prompt(self, init_prompt):
        self.init_prompt = init_prompt

    def getUser(self, username):
        users = []
        with open('users.json', 'r') as file:
            users = json.load(file)

        for user in users:
            if user['name'] == username:
                self.chat_indexes = user['chat_indexes']

    def setUser(self, user):
        self.curUser = user
        self.getUser(user)

    def getChatHistory(self, n):
        indexes = []
        self.getUser(self.curUser)
        chat = ""
        if (len(self.chat_indexes) > 0):
            for index in self.chat_indexes[-n:]:
                indexes.append(f'thought-{index}')

            history = self.memory.fetch(ids=indexes, namespace=THOUGHTS)

            vectors = history['vectors']
            for index in indexes:
                chat += vectors[index]['metadata']['thought_string'] + "\n\n"
        else:
            chat = "This is the user's first time talking to you"

        return chat

    def updateChatIndex(self, username, idx):
        users = []
        user_present = False
        with open('users.json', 'r') as file:
            users = json.load(file)

        for user in users:
            if user['name'] == username:
                user['chat_indexes'].append(idx)
                user_present = True

        if (not user_present):
            users.append({"name": username, "chat_indexes": [idx]})

        with open('users.json', 'w') as file:
            users = json.dump(users, file)

    # Clears all memory and resets agent
    def clearMemory(self):
        self.memory.delete(deleteAll='true', namespace='Thoughts')

        with open('memory_count.yaml', 'w') as f:
            f.write("count: '0'")

        with open('users.json', 'w') as f:
            f.write('[{"name": "user", "chat_indexes": []}]')

    # Look at the memory of a single user
    def viewMemory(self, username):
        users = []
        idx = []
        with open('users.json', 'r') as file:
            users = json.load(file)

        for user in users:
            if user['name'] == username:
                idx = user['chat_indexes']

        for i in range(0, len(idx)):
            idx[i] = 'thought-'+str(idx[i])

        history = self.memory.fetch(ids=idx, namespace=THOUGHTS)
        vectors = history['vectors']
        for id in idx:
            print(vectors[f"{id}"]['metadata']['thought_string'] + "\n\n")

    def createIndex(self, table_name=None):
        # Create Pinecone index
        if (table_name):
            self.table_name = table_name

        if (self.table_name == None):
            return

        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Give memory
        self.memory = pinecone.Index(self.table_name)

    # Adds new Memory to agent, types are: THOUGHTS, ACTIONS, QUERIES, INFORMATION

    def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
            yaml.dump({'count': str(f"{self.thought_id_count}")}, f)

        if thought_type == INFORMATION:
            new_thought = "This is information fed to you by the user:\n" + new_thought
        elif thought_type == QUERIES:
            new_thought = f"{self.curUser}: " + new_thought
        elif thought_type == THOUGHTS:
            # Not needed since already in prompts.yaml
            # new_thought = "You have previously thought:\n" + new_thought
            pass
        elif thought_type == ACTIONS:
            # Not needed since already in prompts.yaml as external thought memory
            pass

        vector = get_ada_embedding(new_thought)
        upsert_response = self.memory.upsert(
            vectors=[
                {
                    'id': f"thought-{self.thought_id_count}",
                    'values': vector,
                    'metadata':
                    {"thought_string": new_thought}
                }],
            namespace=thought_type,
        )

        self.thought_id_count += 1

    # Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
    def internalThought(self, query) -> str:
        query_embedding = get_ada_embedding(query)

        # Get top k memories from Queries and Thoughts
        query_results = self.memory.query(
            query_embedding, top_k=1, include_metadata=True, namespace=QUERIES)
        thought_results = self.memory.query(
            query_embedding, top_k=1, include_metadata=True, namespace=THOUGHTS)
        results = query_results.matches + thought_results.matches
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        # Get top k memories from Queries and Thoughts
        top_matches = "\n\n".join(
            [(str(item.metadata["thought_string"])) for item in sorted_results])
        # print(top_matches)

        # Give agent memories
        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace(
            "{top_matches}", top_matches).replace("{username}", self.curUser)

        # Give agent conversation history
        chat = self.getChatHistory(2)
        internalThoughtPrompt = internalThoughtPrompt.replace(
            "{conversation_history}", chat)

        if (self.seeThoughts):
            print("------------INTERNAL THOUGHT PROMPT START----------")
            print(internalThoughtPrompt)
            print("------------INTERNAL THOUGHT PROMPT END----------\n")
        # OPENAI CALL: top_matches and query text is used here
        internal_thought = generate(internalThoughtPrompt)

        # Debugging purposes
        # print(internal_thought)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query).replace(
            "{internal_thought}", internal_thought).replace("{username}", self.curUser)

        # Update Memories
        self.updateMemory(internalMemoryPrompt, THOUGHTS)
        return internal_thought, top_matches

    def action(self, query) -> str:
        query_embedding = get_ada_embedding(query)

        # Get top k memories from Queries and Thoughts
        query_results = self.memory.query(
            query_embedding, top_k=1, include_metadata=True, namespace=QUERIES)
        thought_results = self.memory.query(
            query_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
        results = query_results.matches + thought_results.matches
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        # Get top k memories from Queries and Thoughts
        top_matches = "\n\n".join(
            [(str(item.metadata["thought_string"])) for item in sorted_results])

        externalThoughtPrompt = data['external_thought']
        externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query).replace(
            "{top_matches}", top_matches).replace("{username}", self.curUser)

        # Give agent conversation history
        chat = self.getChatHistory(2)
        externalThoughtPrompt = externalThoughtPrompt.replace(
            "{conversation_history}", chat)

        if (self.seeThoughts):
            print("------------EXTERNAL THOUGHT PROMPT START----------")
            print(externalThoughtPrompt)
            print("------------EXTERNAL THOUGHT PROMPT END----------\n")
        # OPENAI CALL: top_matches and query text is used here
        external_thought = generate(externalThoughtPrompt)

        # Add current chat index to user
        self.updateChatIndex(self.curUser, self.thought_id_count)

        # Update Memory
        externalMemoryPrompt = data['external_thought_memory']
        externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query).replace(
            "{external_thought}", external_thought).replace("{username}", self.curUser)
        self.updateMemory(externalMemoryPrompt, THOUGHTS)

        return external_thought

    # Make agent think some information

    def think(self, text) -> str:
        self.updateMemory(text, THOUGHTS)

    # Make agent read some information

    def read(self, text) -> str:
        texts = text_splitter.split_text(text)
        vectors = []
        for t in texts:
            t = "This is information fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id': f"thought-{self.thought_id_count}",
                'values': vector,
                'metadata':
                    {"thought_string": t,
                     }
            })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
            vectors,
            namespace=INFORMATION,
        )
    # Make agent read a document

    def readDoc(self, text) -> str:
        texts = text_splitter.split_text(read_txtFile(text))
        vectors = []
        for t in texts:
            t = "This is a document fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id': f"thought-{self.thought_id_count}",
                'values': vector,
                'metadata':
                    {"thought_string": t,
                     }
            })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
            vectors,
            namespace=INFORMATION,
        )
