from textwrap import dedent

from langchain.prompts import PromptTemplate

from .enums import LLMFamily, PromptType


class HybridRagPrompt:
    REFORM_QUESTION = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                Below is a chat history of a current conversation and a follow up input. If the follow up input
                provided by user related to the chat history, rephrase the input into a standalone question. Otherwise,
                return the input as is for the standalone question.

                FOCUS ON REFORMING THE QUESTION. DO NOT INCLUDE EXTRA PREAMBLE OR EXTRA EXPLANATION

                Example:
                ------- BEGIN EXAMPLE -------
                Chat History:

                Human: Do you know about Cristiano Ronaldo?
                AI: Yes, I do. Cristiano Ronaldo or CR7 is one of the best footballers in the world. He plays for
                multiple teams including: Sporting, Manchester United, Real Madrid, Juventus and has won multiple
                championships with them.
                Human: How many goals did he score?
                AI: Ronaldo managed to score almost 900 goals throughout his career

                Follow Up Input: How about Messi?

                Standalone Question: Do you know about Messi and how many goals did he score?
                ------- END EXAMPLE -------

                Chat History:

                {chat_history}

                Follow Up Input: {question}

                Standalone Question:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ),
            input_variables=["question", "chat_history"],
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """Below is a chat history of a current conversation and a follow up input. If the follow up input
                provided by user related to the chat history, rephrase the input into a standalone question. Otherwise,
                return the input as is for the standalone question.

                FOCUS ON REFORMING THE QUESTION. DO NOT INCLUDE EXTRA PREAMBLE OR EXTRA EXPLANATION

                Example:
                ------- BEGIN EXAMPLE -------
                Chat History:

                Human: Do you know about Cristiano Ronaldo?
                AI: Yes, I do. Cristiano Ronaldo or CR7 is one of the best footballers in the world. He plays for
                multiple teams including: Sporting, Manchester United, Real Madrid, Juventus and has won multiple
                championships with them.
                Human: How many goals did he score?
                AI: Ronaldo managed to score almost 900 goals throughout his career

                Follow Up Input: How about Messi?

                Standalone Question: Do you know about Messi and how many goals did he score?
                ------- END EXAMPLE -------

                Chat History:

                {chat_history}

                Follow Up Input: {question}

                Standalone Question:"""
            ),
            input_variables=["question", "chat_history"],
        ),
    }

    SUMMARY_CONVERSATION = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                Progressively summarize the lines of conversation provided, adding onto the previous summary returning a
                new summary.

                FOCUS ON ANSWERING THE QUESTION. ONLY RETURN THE "New summary", DO NOT INCLUDE EXTRA PREAMBLE OR
                EXPLANATION.

                Here is an example:
                \n------- BEGIN EXAMPLE -------\n
                Current summary:
                The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a
                force for good.

                New lines of conversation:
                Human: Why do you think artificial intelligence is a force for good?
                AI: Because artificial intelligence will help humans reach their full potential.

                New summary:
                The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a
                force for good because it will help humans reach their full potential.
                \n------- END EXAMPLE -------\n

                Current summary:
                {summary}

                New lines of conversation:
                {new_lines}

                New summary:
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ),
            input_variables=["summary", "new_lines"],
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """Progressively summarize the lines of conversation provided, adding onto the previous summary
                returning a new summary.

                FOCUS ON ANSWERING THE QUESTION. ONLY RETURN THE "New summary", DO NOT INCLUDE EXTRA PREAMBLE OR
                EXPLANATION.

                Here is an example:
                \n------- BEGIN EXAMPLE -------\n
                Current summary:
                The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is
                a force for good.

                New lines of conversation:
                Human: Why do you think artificial intelligence is a force for good?
                AI: Because artificial intelligence will help humans reach their full potential.

                New summary:
                The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is
                a force for good because it will help humans reach their full potential.
                \n------- END EXAMPLE -------\n

                Current summary:
                {summary}

                New lines of conversation:
                {new_lines}

                New summary:
                """
            ),
            input_variables=["summary", "new_lines"],
        ),
    }

    RETRIEVER_GRADER = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a grader assessing relevance of a retrieved document to a user question. If the document
                contains keywords related to the user question, grade it as relevant. It does not need to be a
                stringent test. The goal is to filter out erroneous retrievals.

                Here is the retrieved document:
                \n------- BEGIN DOCUMENT -------\n
                {document}
                \n------- END DOCUMENT -------\n

                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                {question}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ),
            input_variables=["document", "question"],
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """You are a grader assessing relevance of a retrieved document to a user question. If the document
                contains  keywords related to the user question, grade it as relevant. It does not need to be a
                stringent test. The goal is to filter out erroneous retrievals.

                Here is the retrieved document:
                \n------- BEGIN DOCUMENT -------\n
                {document}
                \n------- END DOCUMENT -------\n

                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE

                Question:
                {question}

                Answer:"""
            ),
            input_variables=["document", "question"],
        ),
    }

    GENERATE_RAG_ANSWER = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for
                question-answering tasks. You are able to generate human-like text based on the input you receives,
                creating a natural-sounding conversations and provide responses that are coherent and relevant to the
                topic at hand.

                Use the following pieces of retrieved context to answer the question. If you don't know the answer,
                just say that you don't know. Consider the given context as your own knowledge, do not mention it in
                the answer. Try to give the answer as detailed as possible

                Context:
                \n------- BEGIN CONTEXT -------\n
                {context}

                {relationships}
                \n------- END CONTEXT -------\n

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                {question}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ),
            input_variables=["context", "relationships", "question"],
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """You are an assistant for question-answering tasks. You are able to generate human-like text based on
                the input you receives, creating a natural-sounding conversations and provide responses that are
                coherent and relevant to the topic at hand. Use the following pieces of retrieved context to answer the
                question. If you don't know the answer, just say that you don't know. Consider the given context as your
                own knowledge, do not mention it in the answer. Try to give the answer as detailed as possible

                Context:
                \n------- BEGIN CONTEXT -------\n
                {context}

                {relationships}
                \n------- END CONTEXT -------\n

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE

                Question:
                {question}

                Answer:"""
            ),
            input_variables=["context", "relationships", "question"],
        ),
    }

    GENERATE_REGULAR_ANSWER = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for
                question-answering tasks. You are able to generate human-like text based on the input you receives,
                creating a natural-sounding conversations and provide responses that are coherent and relevant to the
                topic at hand.

                Use the following summary of chat history to get more context if needed. If you don't know the answer,
                just say that you don't know. Try to give the answer as detailed as possible

                Chat history summary:
                \n------- BEGIN CHAT HISTORY SUMMARY -------\n
                {summary}
                \n------- END CHAT HISTORY SUMMARY -------\n

                Chat history:
                \n------- BEGIN CHAT HISTORY -------\n
                {history}
                \n------- END CHAT HISTORY -------\n

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                {question}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ),
            input_variables=["summary", "history", "question"],
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """You are an assistant for
                question-answering tasks. You are able to generate human-like text based on the input you receives,
                creating a natural-sounding conversations and provide responses that are coherent and relevant to the
                topic at hand.

                Use the following summary of chat history to get more context if needed. If you don't know the answer,
                just say that you don't know. Try to give the answer as detailed as possible

                Chat history summary:
                \n------- BEGIN CHAT HISTORY SUMMARY -------\n
                {summary}
                \n------- END CHAT HISTORY SUMMARY -------\n

                Chat history:
                \n------- BEGIN CHAT HISTORY -------\n
                {history}
                \n------- END CHAT HISTORY -------\n

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE

                Question:
                {question}

                Answer:"""
            ),
            input_variables=["summary", "history", "question"],
        ),
    }

    HALLUCINATION_GRADER = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
                input is grounded in / supported by the given context and relationships between related entities.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Here is the input:
                \n------- BEGIN INPUT -------\n
                {generation}
                \n------- END INPUT -------\n

                Here are the given context:
                \n------- BEGIN CONTEXT -------\n
                {documents}

                {relationships}
                \n------- END CONTEXT -------\n

                Give a binary score 'yes' or 'no' score to indicate whether the generation is grounded in / supported
                by the context. Provide the binary score as a JSON with a single key 'score' and no preamble or
                explanation.

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ),
            input_variables=["generation", "documents", "relationships"],
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """You are a grader assessing whether an input is grounded in / supported by the given context and
                relationships between related entities.

                Here is the input:
                \n------- BEGIN INPUT -------\n
                {generation}
                \n------- END INPUT -------\n

                Here are the given context:
                \n------- BEGIN CONTEXT -------\n
                {documents}

                {relationships}
                \n------- END CONTEXT -------\n

                Give a binary score 'yes' or 'no' score to indicate whether the input is grounded in / supported
                by the context. Provide the binary score as a JSON with a single key 'score' and no preamble or
                explanation.

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE

                Answer:"""
            ),
            input_variables=["generation", "documents", "relationships"],
        ),
    }

    ANSWER_GRADER = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
                input is useful to resolve a question.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Here is the input:
                \n ------- BEGIN INPUT ------- \n
                {generation}
                \n ------- END INPUT ------- \n
                Here is the question:
                \n ------- BEGIN QUESTION ------- \n
                {question}
                \n ------- END QUESTION ------- \n

                Give a binary score 'yes' or 'no' score to indicate whether the input is is useful to resolve the
                given question. Provide the binary score as a JSON with a single key 'score' and no preamble or
                explanation.

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ),
            input_variables=["generation", "question"],
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """You are a grader assessing whether an input is useful to resolve a question.

                Here is the input:
                \n ------- BEGIN INPUT ------- \n
                {generation}
                \n ------- END INPUT ------- \n
                Here is the question:
                \n ------- BEGIN QUESTION ------- \n
                {question}
                \n ------- END QUESTION ------- \n

                Give a binary score 'yes' or 'no' score to indicate whether the input is is useful to resolve the
                given question. Provide the binary score as a JSON with a single key 'score' and no preamble or
                explanation.

                FOCUS ON ANSWERING THE QUESTION, DO NOT INCLUDE EXTRA PREAMBLE

                Answer:"""
            ),
            input_variables=["generation", "question"],
        ),
    }

    EXTRACT_ENTITIES = {
        LLMFamily.llama: PromptTemplate(
            template=dedent(
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a professional entity extractor, Your job is to create a list of entities from the given
                question.
                DO NOT CREATE UNNECESSARY TEXT OR PREAMBLE, YOUR RESPONSE MUST BE A SINGLE JSON LIST OF ENTITIES.

                --- Example ---
                question: Who are Elysian Skippers?
                AI: ["Elysian Skippers"]

                question: When did James meet Mary?
                AI: ["James", "Mary"]

                question: Where did they go after that?
                AI: ["James", "Mary"]
                --- End of example ---

                Here is user's question
                question: {question}
                <|eot_id|><|start_header_id|>AI: <|end_header_id|>"""
            )
        ),
        LLMFamily.other: PromptTemplate(
            template=dedent(
                """You are a professional entity extractor, Your job is to create a list of entities from the given
                question.
                DO NOT CREATE UNNECESSARY TEXT OR PREAMBLE, YOUR RESPONSE MUST BE A SINGLE JSON LIST OF ENTITIES.

                --- Example ---
                question: Who are Elysian Skippers?
                AI: ["Elysian Skippers"]

                question: When did James meet Mary?
                AI: ["James", "Mary"]

                question: Where did they go after that?
                AI: ["James", "Mary"]
                --- End of example ---

                Here is user's question
                question: {question}
                AI: """
            )
        ),
    }

    prompt_map = {
        PromptType.REFORM_QUESTION: REFORM_QUESTION,
        PromptType.SUMMARY_CONVERSATION: SUMMARY_CONVERSATION,
        PromptType.RETRIEVER_GRADER: RETRIEVER_GRADER,
        PromptType.GENERATE_RAG_ANSWER: GENERATE_RAG_ANSWER,
        PromptType.GENERATE_REGULAR_ANSWER: GENERATE_REGULAR_ANSWER,
        PromptType.HALLUCINATION_GRADER: HALLUCINATION_GRADER,
        PromptType.ANSWER_GRADER: ANSWER_GRADER,
        PromptType.EXTRACT_ENTITIES: EXTRACT_ENTITIES,
    }

    @classmethod
    def get_prompt(
        cls, prompt_type: PromptType, llm_family: LLMFamily = LLMFamily.llama
    ):
        return cls.prompt_map.get(prompt_type).get(llm_family)
