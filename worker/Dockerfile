FROM python:3.11.9-slim
WORKDIR /app
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker/*.py /app/
COPY prompts/teacher_prompt.txt /app/prompts/teacher_prompt.txt
CMD ["python", "worker.py"]
