from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts import PromptTemplate
from src.llm_handler import get_llm
import logging

logger = logging.getLogger(__name__)

def create_rag_chain(vectorstore, llm=None):
    """Создание RAG цепочки с кастомным промптом"""
    
    # Кастомный промпт для Claude
    custom_prompt = PromptTemplate.from_template("""
Вы - полезный ассистент, отвечающий на вопросы по предоставленным документам.
Используйте следующие фрагменты контекста, чтобы ответить на вопрос в конце.
Если вы не знаете ответа, просто скажите, что не знаете. Не пытайтесь придумать ответ.

Контекст:
{context}

История разговора:
{chat_history}

Человек: {question}
Ассистент:""")
    
    if llm is None:
        llm = get_llm()
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT
    )
    logger.info("RAG цепочка создана")
    return qa_chain

def format_sources(source_documents):
    """Форматирование источников для отображения"""
    sources = []
    for doc in source_documents:
        source_info = {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        }
        sources.append(source_info)
    return sources