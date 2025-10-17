import asyncio
import requests
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from mcp_use import MCPAgent, MCPClient
import os

# Function to call the REST API
def get_product_details(product_name):
    """Call REST API to get product details by name"""
    try:
        url = f"http://localhost:8080/api/products/name/{product_name}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()  # Assuming the response is in JSON format
        else:
            return f"Error: Unable to fetch product details. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

async def run_memory_chat():
    """Run a chat using MCPAgent's built-in conversation memory."""

    # Load environment variables for API keys
    load_dotenv()

    print("Initializing chat...")

    # Create MCP client and agent with memory enabled
    client = MCPClient.from_config_file("./browser_mcp.json")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    # Create agent with memory_enabled=True
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,  # Enable built-in conversation memory
    )

    print("\n==== Interactive MCP Chat ====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("==============================\n")

    try:
        # Main chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            # Check for clear history command
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            # Check if the user is asking about a product
            if "details about product" in user_input.lower():
                # Extract product name from the user input
                product_name = user_input.lower().replace("give me the details about product", "").strip()
                product_details = get_product_details(product_name)

                # Return the product details
                print("\nAssistant: ", end="", flush=True)
                print(product_details)
                continue

            # Get response from agent
            print("\nAssistant: ", end="", flush=True)
            try:
                # Run the agent with the user input (memory handling is automatic)
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        # Clean up
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())


    
# // Grameen Pitha is a product