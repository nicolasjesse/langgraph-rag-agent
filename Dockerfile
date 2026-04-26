# AWS Lambda image for the LangGraph RAG agent.
#
# Builds a container that ships the agent code + a pre-baked Chroma collection.
# At runtime, lambda_handler.py copies chroma_db to /tmp (the only writable
# directory in Lambda) and exposes handler(event, context).

FROM public.ecr.aws/lambda/python:3.12

# Install Python dependencies. Copy requirements.txt first so Docker can cache
# the pip install layer when only application code changes.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Application code.
COPY agent.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Pre-baked vector store. Re-run scrape_corpus.py + ingest.py locally to refresh.
COPY chroma_db ${LAMBDA_TASK_ROOT}/chroma_db

# Lambda invokes <module>.<function>.
CMD ["lambda_handler.handler"]
