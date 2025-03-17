from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = (
    "I want you to extract specific content from the following text content : {dom_content}."
    "Extract only the information that directly matches the provided description: {parse_description}" 
)

model = OllamaLLM(model="llama3")

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    parsed_results = []

    for i, chunk in enumerate(dom_chunks,start=1):
        response = chain.invoke({"dom_content": chunk, "parse_description": parse_description})
        print(f"parsed batch {i} of {len(dom_chunks)}")
        parsed_results.append(response)

    return "\n".join(parsed_results)

