from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def get_qa_chain(vector_db, model_name="llama3"):
    """
    Returns the LLM and retriever for use in the answer_question function.
    """
    llm = OllamaLLM(model=model_name)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return llm, retriever

def answer_question(qa_components, query):
    """
    Performs manual retrieval and generation without relying on RetrievalQA.
    """
    llm, retriever = qa_components
    
    # 1. Retrieve relevant chunks
    source_documents = retriever.get_relevant_documents(query)
    
    # 2. Combine document context
    context = "\n\n".join([doc.page_content for doc in source_documents])
    
    # 3. Define the prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 4. Generate the answer
    formatted_prompt = prompt.format(context=context, question=query)
    result = llm.invoke(formatted_prompt)
    
    return result, source_documents
