# LLM Fine-Tuning & Retrieval-Augmented Generation

Building a course-assistant chatbot two ways: by **fine-tuning** a small LLM on generated Q&A pairs, and by **retrieval-augmented generation (RAG)** over lecture transcripts.

Built as a course project for **Data Management for Data Science (CS 639), UW–Madison, Spring 2026**. The project started from a course-provided notebook. In the notebook, each code cell is labeled **[My implementation]** or **[Course-provided]**.

## Pipeline

1. **Base model:** Llama-3.2-1B-Instruct, loaded with 4-bit NF4 quantization to fit on a free Colab T4 GPU.
2. **Fine-tuning:** lecture transcripts are cleaned and chunked; Qwen2.5-7B-Instruct generates exam-style Q&A pairs from each chunk; Llama is fine-tuned on them with LoRA and evaluated by perplexity.
3. **RAG:** transcripts are indexed in Elasticsearch with BM25 using two chunking strategies (fixed-size token windows vs. 5-sentence groups), compared by Precision@3, and served through a Haystack pipeline in a Streamlit chat app.

## My contributions

- Transcript cleaning and fixed-size, overlapping chunking used to build the fine-tuning dataset.
- Test prompts for comparing the model before and after fine-tuning, and against RAG.
- Sentence-level chunking (`sentence_chunks`).
- Indexing both chunking strategies into Elasticsearch.
- Precision@3 retrieval evaluation comparing the two chunking strategies.

## Results

**Retrieval quality (Precision@3, 5 exam-style questions, relevance labeled by hand):**

| Question | Fixed-size chunks | Sentence chunks |
| --- | --- | --- |
| What are window functions in SQL? | 0.67 | **1.00** |
| What is relational algebra? | 0.33 | **1.00** |
| What is a SQL subquery and when would you use one? | **0.67** | 0.33 |
| How does MongoDB handle geospatial queries? | 0.00 | 0.00 |
| Elasticsearch vs. a relational database? | 0.00 | 0.00 |
| **Average** | 0.33 | **0.47** |

Sentence-level chunking retrieved better on average because it keeps whole sentences together, so key terms aren't split across chunks. Both strategies missed topics the lectures barely covered.

**Fine-tuning:** the LoRA-tuned model had lower perplexity on held-out Q&A pairs than the base model, which matched its more course-specific answers.

**Fine-tuning vs. RAG:** RAG gave more grounded answers on topics the lectures covered well. Both approaches still hallucinated on topics outside the course, such as naming NoSQL databases the course never taught.

## Tech

Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, TRL, bitsandbytes, Haystack, Elasticsearch, Streamlit.

## Notes

- The lecture transcripts are course material and are not included.
- No API keys are stored here; the notebook prompts for credentials at runtime.
